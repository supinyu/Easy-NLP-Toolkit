# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 16:39
# @Author  : supinyu
# @File    : acclerator_train_lora.py.py
import logging
import os
import sys
import time
import tqdm
import json
import torch
import numpy as np
import loralib as lora
from datasets import load_dataset
from peft import LoraConfig

from arguments import ModelArguments, DataTrainingArguments, FineTuneArguments
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, HfArgumentParser, set_seed, DataCollatorForSeq2Seq
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
log_name = __name__


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FineTuneArguments))
    writer = SummaryWriter()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_name, 'w+', encoding='utf-8'), ],
    )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load dataset
    data_files = {}
    data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = AutoModel.from_pretrained(model_args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_dir, trust_remote_code=True)

    config = LoraConfig(r=training_args.lora_rank,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    accumulate_step = 8
    mixed_precision = 'bf16'
    warm_up_ratio = 0.1
    lr = training_args.learning_rate
    batch_size = training_args.per_device_train_batch_size
    num_epoch = training_args.num_train_epochs
    deepspeed_plugin = DeepSpeedPlugin(gradient_accumulation_steps=accumulate_step)
    accelerator = Accelerator(mixed_precision=mixed_precision, deepspeed_plugin=deepspeed_plugin,
                              log_with="tensorboard",
                              project_dir='runs/')
    device = accelerator.device

    accelerator.wait_for_everyone()

    model.use_cache = False

    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length
        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    accelerator.print('Start to process data')

    if "train" not in raw_datasets:
        raise ValueError("--do_train requires a train dataset")
    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
    train_dataset = train_dataset.select(range(max_train_samples))
    column_names = train_dataset.column_names
    with accelerator.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False
    )

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=data_collator, shuffle=True, batch_size=batch_size)

    accelerator.wait_for_everyone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(len(train_dataset) / accumulate_step * warm_up_ratio),
        num_training_steps=(len(train_dataset) // accumulate_step * num_epoch),
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                           lr_scheduler)
    model.to(device).train()

    accelerator.init_trackers(project_name="accelerator_train_lora")

    total_effective_step = 0

    effective_step = 0

    for epoch in range(num_epoch):

        batch_loss = 0
        effective_step = 0
        for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):
            outputs = model(**batch)
            loss_d = outputs.loss.detach().cpu().float().item()
            batch_loss += loss_d
            loss = outputs.loss / accumulate_step
            accelerator.backward(loss)
            if (step + 1) % accumulate_step == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                effective_step += 1
                gathered_batch_loss = accelerator.gather((torch.tensor(batch_loss, device=device)))
                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train_loss": gathered_batch_loss.mean().item() / accumulate_step,
                            "epoch": epoch,
                        },
                        step=total_effective_step + effective_step,
                    )

                t.set_description(f"loss: {gathered_batch_loss.mean().item() / accumulate_step}")
                batch_loss = 0

        accelerator.wait_for_everyone()

        total_effective_step += effective_step

        if accelerator.is_main_process:
            model_id = f"finetune_{epoch}"
            os.makedirs(f'saved/{model_id}', exist_ok=True)
            accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)),
                             f'saved/{model_id}/{model_id}_epoch_{epoch}.pt')

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
