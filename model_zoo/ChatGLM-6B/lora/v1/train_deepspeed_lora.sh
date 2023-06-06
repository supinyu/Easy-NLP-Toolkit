MODEL_PATH=/disk/nlp_info/LLM_model/ChatGLM-6B
DATA_PATH=/data/supinyu/github/Easy-NLP-Toolkit/datasets/auto_knowledage_extract_cat

CUDA_VISIBLE_DEVICES=0 deepspeed deepspeed_finetuning_lora.py \
  --train_path $DATA_PATH/auto_knowledge_car_llm_train.json \
  --model_dir $MODEL_PATH\
  --num_train_epochs 5 \
  --train_batch_size 1 \
  --lora_r 8
