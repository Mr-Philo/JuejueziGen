python ms_train_reward_model.py \
    --train_path "data/juejuezi_datasets" \
    --save_dir "checkpoints/ms_reward_model/juejuezi_model" \
    --img_log_dir "logs/ms_reward_model/juejuezi_model" \
    --img_log_name "MindSpore Reward Model" \
    --batch_size 32 \
    --max_seq_len 128 \
    --learning_rate 1e-5 \
    --valid_steps 50 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --device "cuda:0"