# -*- coding: utf-8 -*-
# @Time    : 2023/5/22 10:13
# @Author  : supinyu
# @File    : accelerator_predict_lora.py.py



# -*- coding: utf-8 -*-
# @Time    : 2023/5/10 18:28
# @Author  : supinyu
# @File    : deepspeed_predict_lore.py
import torch
import json
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from tqdm import tqdm
import time
import os
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/spo_1.json', type=str, help='')
    parser.add_argument('--model_dir',
                        default="/data/work/lcong/public_model_path/ChatGLM-6B/", type=str,
                        help='')
    parser.add_argument('--lora_model_dir',
                        default="/data/work/lcong/ChatGPT/ChatGLM-Finetuning/output_dir_lora/global_step-3600/", type=str,
                        help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    parser.add_argument('--lora_rank', type=int, default=8, help='')
    return parser.parse_args()


def main():
    args = set_args()
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    config = LoraConfig(r=args.lora_rank,
                        lora_alpha=32,
                        target_modules=["query_key_value"],
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=True)

    model = get_peft_model(model, config)
    model.load_state_dict(torch.load(args.lora_model_dir), strict=False)
    model.half().cuda()
    model.eval()
    save_data = []
    f1 = 0.0
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    eval_index = 0
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(args.prompt_text)

                if len(src_tokens) > args.max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:args.max_src_len - len(prompt_tokens)]

                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_ids = tokenizer.encode("帮我写个快排算法")

                input_ids = torch.tensor([input_ids]).to("cuda")
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }
                response = model.generate_one(input_ids, **generation_kwargs)
                eval_index = eval_index + 1
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    res.append(r)
                pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                real_res = sample["completion"].split("\n")
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                save_data.append(
                    {"text": sample["text"], "ori_answer": sample["completion"], "gen_answer": res[0], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print("total_fl_score: {}".format(f1 / eval_index))
    save_path = os.path.join(args.lora_model_dir, "ft_pt_answer.json")
    fin = open(save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()


if __name__ == '__main__':
    main()