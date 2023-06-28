MODEL_PATH=/disk/nlp_info/nlp_model/bert-base-chinese

deepspeed --include="localhost:1,2" train_gpt2_trainer.py \
  --train_file /disk/nlp_info/MLM_xueqiu_data/raw_data/20220227_all_text_gpt_train.json \
  --validation_file   /disk/nlp_info/MLM_xueqiu_data/raw_data/20220227_all_text_gpt_test.json \
  --model_name_or_path $MODEL_PATH \
  --output_dir news_gpt2 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --context_length  512 \
  --cache_data_dir cache_data \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy steps \
  --eval_steps 5000 \
  --logging_steps 500 \
  --overwrite_cache False \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type cosine \
  --learning_rate 5e-4 \
  --save_steps 1000 \
  --bf16 \
  --deepspeed ds_config.json
