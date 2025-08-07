#!/bin/bash
# accelerate launch --mixed_precision=bf16 train_difix.py \
# --output_dir=/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens_pseudo2 \
# --dataset_path="/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/data/data3.json" \
# --max_train_steps 15000 \
# --resolution=512 --learning_rate 2e-5 \
# --train_batch_size=4 --dataloader_num_workers 0 \
# --enable_xformers_memory_efficient_attention \
# --checkpointing_steps=250 --eval_freq 500 --viz_freq 10000 \
# --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 1.0 --gram_loss_warmup_steps 2000 \
# --tracker_project_name "difix" --tracker_run_name "train" --timestep 199 --mv_unet

# python inference_difix.py \
#   --model_path "/home/xiaoyun/Code/MyCode/Difix3D/models/20250804/model_20001.pkl" \
#   --input_image "/data/hdd/Data/GCI/Final/data/test/micro/lr_test_500" \
#   --ref_image "/data/hdd/Data/GCI/Final/data/test/micro/lr_test_500" \
#   --prompt "remove degradation" \
#   --output_dir "/home/xiaoyun/Code/MyCode/Difix3D/outputs/20250804_lr_test_500_as_ref" \
#   --mv_unet

python inference_difix.py \
  --model_path "/home/xiaoyun/Code/MyCode/Difix3D/models/20250804/model_20001.pkl" \
  --input_image "/data/hdd/Data/GCI/Final/data/test/micro/lr_test_500" \
  --ref_image "/home/xiaoyun/Code/MyCode/Difix3D/outputs/20250804_lr_test_500_as_ref" \
  --prompt "remove degradation" \
  --output_dir "/home/xiaoyun/Code/MyCode/Difix3D/outputs/20250804_lr_test_500_square" \
  --mv_unet

# python evaluate_img.py -i ./outputs/pseudonew_lora_5001_20250804 -r /home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/test/test/micro/gt -o res_pseudonew_lora_5000_20250804.json

