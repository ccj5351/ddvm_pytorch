GPU_ID=$1

DATA_EVAL="kitti sintel"
DATA_EVAL="kitti"
DATA_EVAL="sintel"
DATA_DIR="/home/ccj/code/tpk2scene/datasets"

MACHINE_NAME="rtxa6ks3"

flag=true
#flag=false
if [ "$flag" = true ]; then
    # SRUnet256
    CKPT_PTH="/home/ccj/code/tpk2scene/checkpoints_nfs/pretrained/ddvm_pytorch/autoflow-ImagenUnet/pipeline-900000"
    if [ $DATA_EVAL = "kitti" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/ddvm/imagenunet/kt15_tile"
    elif [ $DATA_EVAL = "sintel" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/ddvm/imagenunet/sintel_tile"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate_diffusers_tile.py \
        --pipeline_path $CKPT_PTH \
        --validation_data $DATA_EVAL \
        --data_dir $DATA_DIR \
        --result_dir $RESULT_DIR \
        --machine_name $MACHINE_NAME \
        --eval_gpu_id $GPU_ID \
        --normalize_range
    #exit
fi

flag=true
#flag=false
RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/ddvm/kt15_tile"
if [ "$flag" = true ]; then
    # RAFT_Unet
    CKPT_PTH="/home/ccj/code/tpk2scene/checkpoints_nfs/pretrained/ddvm_pytorch/autoflow-RAFTUnet/pipeline-305000"
    if [ $DATA_EVAL = "kitti" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/ddvm/raftunet/kt15_tile"
    elif [ $DATA_EVAL = "sintel" ]; then
        RESULT_DIR="/home/ccj/code/tpk2scene/results_nfs/ddvm/raftunet/sintel_tile"
    fi
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 evaluate_diffusers_tile.py \
        --pipeline_path $CKPT_PTH \
        --validation_data $DATA_EVAL \
        --data_dir $DATA_DIR \
        --result_dir $RESULT_DIR \
        --machine_name $MACHINE_NAME \
        --eval_gpu_id $GPU_ID \
        --normalize_range
    #exit
fi