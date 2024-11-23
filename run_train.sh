#accelerate config
## Train ImagenUnet
#accelerate launch --num_processes 8 train_flow_prediction.py --stage 'autoflow' --train_batch_size 16 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-ImagenUnet" --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine'
## Train CorrUnet
#accelerate launch --num_processes 8 train_flow_prediction.py --stage 'autoflow' --train_batch_size 16 --image_size 320 448 --dataloader_num_workers 16 --num_steps 1000000 --save_images_steps 500 --gradient_accumulation_steps 1 --lr_warmup_steps 10000 --use_ema --mixed_precision 'bf16' --prediction_type 'sample' --ddpm_num_steps 64 --checkpointing_steps 10000 --checkpoints_total_limit 5 --output_dir "check_points/autoflow-CorrUnet" --max_flow 400 --learning_rate 1e-4 --adam_weight_decay 0.0001 --it_aug --add_gaussian_noise --normalize_range --lr_scheduler 'cosine' --Unet_type 'RAFT_Unet'


GPU_IDS=$1
PROJ_ROOT="/home/ccj/code/tpk2scene"
DATA_DIR="$PROJ_ROOT/datasets"
ACCELERATE_CONFIG="$PROJ_ROOT/config/huggingface/accelerate/stage2_DeepSpeed.yaml"

MACHINE_NAME="a6ks3"
UNET_TYPE='RAFT_Unet'

DATA_STAGE="autoflow"
#DATA_STAGE="chairs"
BATCH_SIZE=4 # 1-GPU
BATCH_SIZE=12 # 1-GPU
IMAGE_H=320
IMAGE_W=448
NUM_WORKERS=8
NUM_STEPS=1000000 #1000k
SAVE_IMAGES_STEPS=500
LR_WARMUP_STEPS=10000

#{no,fp16,bf16,fp8}
MIXED_PRECISION=fp16

LEARNING_RATE=1e-4
LR_SCHEDULER='cosine'
EXP_NAME="exp01_bl_ddvm_$UNET_TYPE_$MACHINE_NAME"
OUTPUT_DIR="$PROJ_ROOT/experiment-output-nfs/$EXP_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
#exit


#flag=false
flag=true
if [ "$flag" = true ]; then
    IFS=','
    GPU_NUM=$(echo "$GPU_IDS" | tr "$IFS" '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, nproc_per_node=$GPU_NUM"
    #accelerate config
    accelerate launch --config_file $ACCELERATE_CONFIG \
        --num_processes $GPU_NUM \
        train_flow_prediction.py \
        --stage $DATA_STAGE \
        --train_batch_size $BATCH_SIZE \
        --image_size $IMAGE_H $IMAGE_W \
        --dataloader_num_workers $NUM_WORKERS \
        --num_steps $NUM_STEPS \
        --save_images_steps $SAVE_IMAGES_STEPS \
        --gradient_accumulation_steps 1 \
        --lr_warmup_steps $LR_WARMUP_STEPS \
        --mixed_precision $MIXED_PRECISION \
        --prediction_type 'sample' \
        --ddpm_num_steps 64 \
        --checkpointing_steps 10000 \
        --checkpoints_total_limit 5 \
        --output_dir $OUTPUT_DIR \
        --max_flow 400 \
        --learning_rate $LEARNING_RATE \
        --adam_weight_decay 0.0001 \
        --lr_scheduler $LR_SCHEDULER \
        --Unet_type $UNET_TYPE \
        --data_dir $DATA_DIR \
        --it_aug \
        --use_ema \
        --add_gaussian_noise \
        --normalize_range
fi