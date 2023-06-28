# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 17:41
# @Author  : supinyu
# @File    : train_bloom_trainer.py
import logging
from dataclasses import dataclass, field
from typing import Optional

import transformers
from datasets import load_dataset
from transformers import HfArgumentParser, set_seed, TrainingArguments, AutoTokenizer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    cache_data_dir: str = field(
        default="", metadata={"help": "data load cache data dir"}
    )


@dataclass
class FineTuneArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FineTuneArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map='auto',
        trust_remote_code=True
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
        cache_dir=data_args.cache_data_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.print_trainable_parameters()

    IGNORE_INDEX = -100
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
    }
    instruction_column = ""
    input_column = ""
    output_column = ""
    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[instruction_column])):
            instruction, query, answer = examples[instruction_column][i], examples[input_column][i], examples[output_column][i]

            if query == "":
                query = PROMPT_DICT["prompt_no_input"].format(instruction = instruction)
            else:
                query = PROMPT_DICT["prompt_input"].format(instruction = instruction, input= query)




            a_ids = tokenizer.encode(text=query, add_special_tokens=False)
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



if __name__ == "__main__":
    logging.basicConfig(
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
        )
    main()