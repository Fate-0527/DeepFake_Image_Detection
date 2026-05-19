#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dire

EXP_NAME="my_clean_baseline_split"

# [핵심 1] 데이터셋 이름 (폴더명)
DATASETS="clean"
DATASETS_TEST="clean"

# [핵심 2] 루트 경로를 '상위 폴더'로 올려야 함
# DIRE가 이 뒤에 자동으로 '/train/clean'을 붙입니다.
DATA_ROOT="/data1/pilot_dataset_split"

python train.py --gpus 0 \
    --exp_name $EXP_NAME \
    datasets $DATASETS \
    datasets_test $DATASETS_TEST \
    dataset_root $DATA_ROOT \
    batch_size 16 \
    nepoch 30