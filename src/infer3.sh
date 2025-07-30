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

python inference_difix.py \
  --model_path "./outputs/train_multi/checkpoints/model_10001.pkl" \
  --input_image "/data/hdd/Data/GCI/Final/data/test/micro/lr" \
  --ref_image "/home/xiaoyun/Code/RefCode/KAIR/model_zoo/model_w_pseudo/swinir_NMI_x4_psnr_pseudo/sr_results_test_60000" \
  --prompt "remove degradation" \
  --output_dir "./outputs/swinir_retrain_10000"


# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens_pseudo2/checkpoints/model_10001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/pseudo_metamixswin_10000" \
#   --mv_unet

# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens_pseudo2/checkpoints/model_5001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/pseudo_metamixswin_5000" \
#   --mv_unet

# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens_pseudo2/checkpoints/model_3001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/pseudo_metamixswin_3000" \
#   --mv_unet

# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens/checkpoints/model_3001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/metamixswin_3000" \
#   --mv_unet


# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens/checkpoints/model_4001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/metamixswin_4000" \
#   --mv_unet

# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens/checkpoints/model_5001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/metamixswin_5000" \
#   --mv_unet

# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens/checkpoints/model_6001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/metamixswin_6000" \
#   --mv_unet


# python inference_difix.py \
#   --model_path "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/code/Difix3D/Difix3D/outputs/difix/train_mv_metalens/checkpoints/model_7001.pkl" \
#   --input_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/micro_sensor/dataset_tog/test_all/lr" \
#   --ref_image "/home/bml/storage/mnt/v-7c231cc5f5054b0a/org/data/NMI/SwinIR/swinir_NMI_x4_psnr_v2/sr_results_test_iter_20000" \
#   --prompt "remove degradation" \
#   --output_dir "./outputs/metamixswin_7000" \
#   --mv_unet
