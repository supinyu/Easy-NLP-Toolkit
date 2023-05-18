# -*- coding: utf-8 -*-
# @Time    : 2023/5/9 17:21
# @Author  : supinyu
# @File    : data_extract.py.py
import json
import random

from datasets import load_dataset


def data_util(file_input):
    data_list = []
    with open(file_input, "r") as f:
        for line in f:
            lines = json.loads(line.strip())
            text = lines["text"]
            spo_data = lines["spo_list"]
            spo_list = []
            for item in spo_data:
                h_name = item["h"]["name"]
                t_name = item["t"]["name"]
                relation = item["relation"]
                spo_list.append(h_name + "_" + relation + "_" + t_name)
            if len(spo_list) > 0:
                data_list.append((text, "\n".join(spo_list)))
    random.shuffle(data_list)
    eval_data = data_list[0:100]
    eval_data_list = [{"text": item[0], "completion": item[1]} for item in eval_data]
    train_data = data_list[100:]
    train_data_list = [{"text": item[0], "completion": item[1]} for item in train_data]

    with open("auto_knowledge_car_llm_train.csv", "w") as fw:
        for item in train_data_list:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open("auto_knowledge_car_llm_eval.csv", "w") as fw:
        for item in eval_data_list:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    data_util("auto_knowledge_car_train.json")
    data_file_dict = {}
    data_extension = "json"
    data_file_dict["train"] = "auto_knowledge_car_llm_train.csv"
    data_file_dict["validation"] = "auto_knowledge_car_llm_eval.csv"
    local_dataset = load_dataset(data_extension, data_files=data_file_dict)
    print(local_dataset)
