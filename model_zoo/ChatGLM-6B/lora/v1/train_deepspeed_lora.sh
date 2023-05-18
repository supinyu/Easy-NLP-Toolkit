MODEL_PATH=/disk/nlp_info/LLM_model/ChatGLM-6B

CUDA_VISIBLE_DEVICES=0 deepspeed deepspeed_finetuning_lora.py \
  --train_path auto_knowledge_car_llm_train.json \
  --model_dir $MODEL_PATH\
  --train_batch_size 8 \
  --num_train_epochs 5 \
  --train_batch_size 2 \
  --lora_r 8
