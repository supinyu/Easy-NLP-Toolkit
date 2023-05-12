# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 13:59
# @Author  : supinyu
# @File    : data_util.py


import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer

pad_token = 3


class GLMDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        max_target_len = max_len - max_src_len - 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as f:
            # data_json = json.loads(f)
            for i, line in enumerate(f):
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(prompt_text)

                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

                tgt_tokens = tokenizer.tokenize(sample["completion"])
                if len(tgt_tokens) > max_target_len:
                    tgt_tokens = tgt_tokens[:max_target_len]
                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

                self.all_data.append(
                    {"text": sample["text"], "completion": sample["completion"], "input_ids": input_ids,
                     "labels": labels})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def data_collator(batch: list) -> dict:
    input_ids_list, labels_list = [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=pad_token)}


def preprocess_function_train(examples):
    pass


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    train_path = "/Users/supinyu/Documents/GitHub/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat/auto_knowledge_car_llm_eval.csv"
    max_len = 768
    max_src_len = 450
    prompt_text = ""
    train_dataset = GLMDataSet(train_path, tokenizer, max_len, max_src_len, prompt_text)
    print(train_dataset[0])
