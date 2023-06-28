MODEL_PATH=/disk/nlp_info/LLM_model/ChatGLM-6B

CUDA_VISIBLE_DEVICES=2 python deepspeed_predict_lora_goat.py \
  --test_path /disk/nlp_info/LLM_dataset/school_math_0.25M_goat_test.json \
  --model_dir $MODEL_PATH\
  --lora_model_dir /data/supinyu/github/Easy-NLP-Toolkit/model_zoo/ChatGLM-6B/lora/v2/lora_goat_output/checkpoint-59500 \
  --device "0"
