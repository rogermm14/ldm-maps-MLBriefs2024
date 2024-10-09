dataset_path="/home/ubuntu/projects/phase-iv-ai/maps"
output_dir="/media/share/Datasets/diffusers_out/$exp_id"
logs_dir="/media/share/Datasets/diffusers_out/$exp_id"
num_epochs=405
learning_rate="1e-4"
num_inference_steps=1000
bsz=8
save_freq=50 # in epochs
warmup_steps=500
ckpt_steps=5000
resolution=256

# MODEL 2 - Simple conditional synthesis where condition image is encoded 
exp_id="MLBriefs24_1_conditional_vanilla_encode_mask"
accelerate launch train_maps.py \
  --dataset_name=$dataset_path/train \
  --val_dataset_name=$dataset_path/val \
  --resolution=$resolution \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=$learning_rate \
  --lr_warmup_steps=$warmup_steps \
  --checkpointing_steps=$ckpt_steps \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no \
  --encode_cond

exit

# MODEL 3 - Simple conditional synthesis where condition image is downsampled
exp_id="MLBriefs24_2_conditional_vanilla_downsample_mask"
accelerate launch train_maps.py \
  --dataset_name=$dataset_path/train \
  --val_dataset_name=$dataset_path/val \
  --resolution=$resolution \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=$learning_rate \
  --lr_warmup_steps=$warmup_steps \
  --checkpointing_steps=$ckpt_steps \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no

# MODEL 1 - Unconditional synthesis
exp_id="MLBriefs24_0_unconditional"
accelerate launch train_maps.py \
  --dataset_name=$dataset_path/train \
  --val_dataset_name=$dataset_path/val \
  --resolution=$resolution \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=$learning_rate \
  --lr_warmup_steps=$warmup_steps \
  --checkpointing_steps=$ckpt_steps \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no \
  --unconditional

exit



bsz=4

# MODEL 4 - Cross-attention conditional synthesis 
exp_id="MLBriefs24_4_conditional_CA_downsampledmask"
accelerate launch train_maps.py \
  --dataset_name=$dataset_path/train \
  --val_dataset_name=$dataset_path/val \
  --resolution=$resolution \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=$learning_rate \
  --lr_warmup_steps=$warmup_steps \
  --checkpointing_steps=$ckpt_steps \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no \
  --crossattention


exp_id="MLBriefs24_5_conditional_CA_encodedmask"
accelerate launch train_maps.py \
  --dataset_name=$dataset_path/train \
  --val_dataset_name=$dataset_path/val \
  --resolution=$resolution \
  --output_dir="/media/share/Datasets/diffusers_out/$exp_id" \
  --logging_dir=$logs_dir \
  --train_batch_size=$bsz \
  --num_epochs=$num_epochs \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=$learning_rate \
  --lr_warmup_steps=$warmup_steps \
  --checkpointing_steps=$ckpt_steps \
  --save_images_epochs=$save_freq \
  --save_model_epochs=$save_freq \
  --ddpm_num_inference_steps=$num_inference_steps \
  --mixed_precision=no \
  --encode_cond \
  --crossattention

exit


