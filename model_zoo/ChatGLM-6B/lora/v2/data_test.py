# -*- coding: utf-8 -*-
# @Time    : 2023/5/19 09:59
# @Author  : supinyu
# @File    : data_test.py.py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
train_path = "/Users/supinyu/Documents/GitHub/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat/auto_knowledge_car_llm_eval.json"

max_source_length = 450
max_target_length = 318
prompt_column = "text"
response_column = "completion"
history_column  = None
prefix = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："

ignore_pad_token_for_loss = True
ignore_pad_token_for_loss = -100

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
            if ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)

    return model_inputs


data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        label_pad_token_id=ignore_pad_token_for_loss,
        pad_to_multiple_of=None,
        padding=False
    )
data_files = {}
data_files["train"] = "/Users/supinyu/Documents/GitHub/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat/auto_knowledge_car_llm_eval.json"

extension = "json"

raw_datasets = load_dataset(
        extension,
        data_files=data_files,
)

train_dataset = raw_datasets["train"].map(
            preprocess_function_train,
            batched=True,
            num_proc=1,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

print(train_dataset[0])