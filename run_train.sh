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
BATCH_SIZE=8 # 1-GPU
#BATCH_SIZE=12 # 6-GPU
#BATCH_SIZE=16 # 8-GPU
IMAGE_H=320
IMAGE_W=448
NUM_WORKERS=8
NUM_STEPS=1000000 #1000k
SAVE_IMAGES_STEPS=500
LR_WARMUP_STEPS=10000
#CHECKPOINTING_STEPS=10000
CHECKPOINTING_STEPS=1000
CHECKPOINTS_TOTAL_LIMIT=10

#{no,fp16,bf16,fp8}
MIXED_PRECISION=fp16

IFS=','
GPU_NUM=$(echo "$GPU_IDS" | tr "$IFS" '\n' | wc -l)

LEARNING_RATE=1e-4
LR_SCHEDULER='cosine'
EXP_NAME="exp02_bl_ddvm_$UNET_TYPE"
EXP_NAME="${EXP_NAME}_bs${BATCH_SIZE}Gn${GPU_NUM}_${MACHINE_NAME}"
OUTPUT_DIR="$PROJ_ROOT/experiment-output-nfs/$EXP_NAME"


RESUME_FROM_CHECKPOINT="experiment-output-nfs/exp01_bl_ddvm_RAFT_Unet_bs12Gn6_a6ks3-2024-11-25_23:56:56"
#RESUME_FROM_CHECKPOINT=""

# if resume with latest checkpoints, have to specify your EXP_NAME
# RESUME_FROM_CHECKPOINT="latest"
# EXP_NAME="exp01_bl_ddvm_RAFT_Unet_bs12Gn6_a6ks3-2024-11-25_23:56:56"
# OUTPUT_DIR="$PROJ_ROOT/experiment-output-nfs/$EXP_NAME"


#RESUME_FROM_MODEL_ONLY="exp01_bl_ddvm_RAFT_Unet_bs12Gn6_a6ks3-2024-11-25_23:56:56/checkpoint-12000"
#RESUME_FROM_MODEL_ONLY="$PROJ_ROOT/experiment-output-nfs/$RESUME_FROM_MODEL_ONLY"
#RESUME_FROM_CHECKPOINT=""

FINETUNE_FROM_MODEL_ONLY="exp01_bl_ddvm_RAFT_Unet_bs12Gn6_a6ks3-2024-11-25_23:56:56/checkpoint-12000"
FINETUNE_FROM_MODEL_ONLY="$PROJ_ROOT/experiment-output-nfs/$FINETUNE_FROM_MODEL_ONLY"
RESUME_FROM_CHECKPOINT="none"
RESUME_FROM_MODEL_ONLY="none"

# if not resume, we add random (i.e., cur time) to output dir;
# ow, we use 
if [ "$RESUME_FROM_CHECKPOINT" == "" && "$RESUME_FROM_MODEL_ONLY" != "" ]; then
    # Get the current time in the desired format
    #CURRENT_TIME=$(date '+%Y-%m-%d_%H:%M:%S')
    CURRENT_TIME=$(date '+%Y-%m-%d')
    OUTPUT_DIR="$OUTPUT_DIR-$CURRENT_TIME"
    echo "Training from scratch ..."
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"

elif [ "$RESUME_FROM_CHECKPOINT" = "latest" ]; then
    echo "Resuming training using latest checkpoint ... "
    echo "Please make sure your OUTPUT_DIR is correct ..."

    # Prompt the user for confirmation
    read -p "It was $OUTPUT_DIR. Do you want to continue? (y/n): " response

    # Convert response to lowercase for consistency
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
    if [[ "$response" == "y" || "$response" == "yes" ]]; then
        echo "You chose Yes. Proceeding..."
    else
        echo "You chose No. Exiting..."
        exit 1
    fi
    echo "[***] OUTPUT_DIR=$OUTPUT_DIR"
    echo "[***] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"

elif [[ "$RESUME_FROM_MODEL_ONLY" != "" || "$FINETUNE_FROM_MODEL_ONLY" != "" ]]; then
    echo "Resuming/Finetune training using only model weights ... "
    echo "Please make sure your OUTPUT_DIR is correct ..."
    CURRENT_TIME=$(date '+%Y-%m-%d')
    OUTPUT_DIR="$OUTPUT_DIR-$CURRENT_TIME"
    RESUME_FROM_CHECKPOINT="none"
    echo "Finetuning from model weights ..."

    # Prompt the user for confirmation
    read -p "It was $OUTPUT_DIR. Do you want to continue? (y/n): " response

    # Convert response to lowercase for consistency
    response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
    if [[ "$response" == "y" || "$response" == "yes" ]]; then
        echo "You chose Yes. Proceeding..."
    else
        echo "You chose No. Exiting..."
        exit 1
    fi
    echo "[***] OUTPUT_DIR=$OUTPUT_DIR"
    echo "[***] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
    echo "[***] RESUME_FROM_MODEL_ONLY=$RESUME_FROM_MODEL_ONLY"
    echo "[***] FINETUNE_FROM_MODEL_ONLY=$FINETUNE_FROM_MODEL_ONLY"
else
    echo "Resuming training ... "
    echo "OUTPUT_DIR should be the same as RESUME_FROM_CHECKPOINT ..."
    RESUME_FROM_CHECKPOINT="$PROJ_ROOT/$RESUME_FROM_CHECKPOINT"
    OUTPUT_DIR="$RESUME_FROM_CHECKPOINT"
    echo "[***] OUTPUT_DIR=$OUTPUT_DIR"
    echo "[***] RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT"
fi

#exit

#flag=false
flag=true
if [ "$flag" = true ]; then
    export OMP_NUM_THREADS=1
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
        --checkpointing_steps $CHECKPOINTING_STEPS \
        --checkpoints_total_limit $CHECKPOINTS_TOTAL_LIMIT \
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
        --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
        --resume_from_model_only $RESUME_FROM_MODEL_ONLY \
        --finetune_from_model_only $FINETUNE_FROM_MODEL_ONLY \
        --normalize_range
fi