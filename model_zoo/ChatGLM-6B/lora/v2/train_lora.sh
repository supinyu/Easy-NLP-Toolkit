MODEL_PATH=/disk/nlp_info/LLM_model/ChatGLM-6B
source_prefix_str="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："

CUDA_VISIBLE_DEVICES=2,3 deepspeed deepspeed_finetuning_lora_v2.py \
  --train_file /data/supinyu/github/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat/auto_knowledge_car_llm_train.json \
  --model_name_or_path $MODEL_PATH \
  --output_dir lora_output \
  --prompt_column text \
  --response_column completion \
  --source_prefix source_prefix_str\
  --overwrite_cache \
  --overwrite_output_dir \
  --max_source_length 512 \
  --max_target_length 256 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --logging_steps 50 \
  --save_steps 1000 \
  --lora_rank 8 \
  --fp16 \
  --deepspeed ds_config.json
