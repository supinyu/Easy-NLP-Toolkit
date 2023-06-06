# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 16:39
# @Author  : supinyu
# @File    : acclerator_train_lora.py.py
import argparse
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
from peft import LoraConfig, get_peft_model

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModel, HfArgumentParser, set_seed, DataCollatorForSeq2Seq
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)
log_name = __name__


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='data/spo_0.json', type=str, help='')
    parser.add_argument('--model_name_or_path', default="/data/work/lcong/public_model_path/ChatGLM-6B/", type=str,
                        help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--per_device_train_batch_size', default=2, type=int, help='')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='')
    parser.add_argument('--output_dir', default='output_dir_lora/', type=str, help='')
    parser.add_argument('--max_source_length', type=int, default=768, help='')
    parser.add_argument('--max_target_length', type=int, default=450, help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--lora_rank', type=int, default=8, help='')
    parser.add_argument('--prompt_column', type=str, default="text", help='')
    parser.add_argument('--response_column', type=str, default="response_column", help='')
    parser.add_argument('--source_prefix', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    return parser.parse_args()



args = set_args()
set_seed(args.seed)

mixed_precision = 'bf16'
accumulate_step = 8
LR = args.learning_rate
warm_up_ratio = 0.1
batch_size = args.per_device_train_batch_size
num_epoch = args.num_train_epochs

config = LoraConfig(r=args.lora_rank,
                    lora_alpha=32,
                    target_modules=["query_key_value"],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                    inference_mode=False,
                    )

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
model = get_peft_model(model, config)
model = model.half()

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step,
                          deepspeed_plugin=deepspeed_plugin)
device = accelerator.device

data_files = {}
data_files["train"] = "/data/supinyu/github/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat/github_train.json"
extension = data_files["train"].split(".")[-1]

raw_datasets = load_dataset(
    extension,
    data_files=data_files
)

max_source_length = args.max_source_length
max_target_length = args.max_target_length
prompt_column = args.prompt_column
response_column = args.response_column
history_column = None
prefix = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："

def preprocess_function_train(examples):
    max_seq_length = max_source_length + max_target_length
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

            if len(a_ids) > max_source_length - 1:
                a_ids = a_ids[: max_source_length - 1]

            if len(b_ids) > max_target_length - 2:
                b_ids = b_ids[: max_target_length - 2]

            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if True:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs

train_dataset = raw_datasets["train"].shuffle()
column_names = train_dataset.column_names
with accelerator.main_process_first():
    train_dataset = train_dataset.map(
        preprocess_function_train,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        load_from_cache_file=False,
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

### Training

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * num_epoch),
)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.to(device).train()

for epoch in range(num_epoch):
    total_loss = 0
    for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):
        outputs = model(**batch)
        loss_d = outputs.loss.detach()
        t.set_description(f"loss: {loss_d.cpu().float().numpy()}")
        # epoch_loss_local += loss_d
        loss = outputs.loss / accumulate_step
        accelerator.backward(loss)
        if (step + 1) % accumulate_step == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        peft_model_id = f"finetune_{epoch}"
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), './' + peft_model_id + '.pt')

    accelerator.wait_for_everyone()

