# -*- coding: utf-8 -*-
# @Time    : 2023/6/2 17:52
# @Author  : supinyu
# @File    : train_gpt2_trainer.py

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, set_seed, HfArgumentParser
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

from arguments import DataTrainingArguments, ModelArguments, FineTuneArguments


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FineTuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    raw_datasets = load_dataset("json", data_files={'train': data_args.train_file, 'valid': data_args.validation_file},
                                cache_dir=data_args.cache_data_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens(special_tokens_dict={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>',
                                                      'unk_token': '<|endoftext|>'})

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=training_args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == training_args.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names, num_proc=30
    )

    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=training_args.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id)

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # train_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     per_device_train_batch_size=args.train_batch_size,
    #     per_device_eval_batch_size=args.eval_batch_size,
    #     evaluation_strategy="steps",
    #     eval_steps=5000,
    #     logging_steps=500,
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     num_train_epochs=1,
    #     weight_decay=0.1,
    #     warmup_steps=100,
    #     lr_scheduler_type="cosine",
    #     learning_rate=5e-4,
    #     save_steps=5000,
    #     bf16=True,
    #     push_to_hub=False,
    #     deepspeed=args.deepspeed if args.deepspeed is not None else None
    # )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()


if __name__ == "__main__":
    main()
