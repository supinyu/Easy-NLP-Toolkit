MODEL_PATH=/disk/nlp_info/LLM_model/ChatGLM-6B

CUDA_VISIBLE_DEVICES=2 python deepspeed_finetuning_lora.py \
  --test_path auto_knowledge_car_llm_eval.csv \
  --model_dir $MODEL_PATH\
  --lora_model_dir /data/supinyu/github/LLM_notes/ChatGLM-6B/output_dir_lora/global_step-6845 \
  --device "0"
