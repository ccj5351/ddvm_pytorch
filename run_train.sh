#accelerate config
## Train ImagenUnet
#accelerate launch --num_processes 8 train_flow_prediction.py --stage 'autoflow' --train_batch_size 16 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-ImagenUnet" --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine'
## Train CorrUnet
#accelerate launch --num_processes 8 train_flow_prediction.py --stage 'autoflow' --train_batch_size 16 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-CorrUnet" --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine' --Unet_type 'RAFT_Unet'


GPU_IDS=$1

DATA_EVAL="kitti sintel"
DATA_EVAL="kitti"
#DATA_EVAL="sintel"
DATA_DIR="/home/ccj/code/tpk2scene/datasets"

MACHINE_NAME="rtxa6ks3"

UNET_TYPE='RAFT_Unet'
STAGE="autoflow"
EXP_NAME="exp01_bl_ddvm_${UNET_TYPE}"

#flag=false
flag=true
if [ "$flag" = true ]; then
    IFS=','
    GPU_NUM=$(echo "$GPU_IDS" | tr "$IFS" '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc_per_node=$GPU_NUM"
    accelerate launch --num_processes $GPU_NUM train_flow_prediction.py \
        --stage 'autoflow' \
        --train_batch_size 16 \
        --image_size 320 448 \
        --dataloader_num_workers 16 \
        --num_steps 1000000 \
        --save_images_steps 500 \
        --gradient_accumulation_steps 1 \
        --lr_warmup_steps 10000 \
        --use_ema \
        --mixed_precision 'bf16' \
        --prediction_type 'sample' \
        --ddpm_num_steps 64 \
        --checkpointing_steps 10000 \
        --checkpoints_total_limit 5 \
        --output_dir "checkpoints_nfs/autoflow-CorrUnet" \
        --max_flow 400 --learning_rate 1e-4 \
        --adam_weight_decay 0.0001 \
        --it_aug --add_gaussian_noise \
        --normalize_range \
        --lr_scheduler 'cosine' \
        --Unet_type $UNET_TYPE
fi