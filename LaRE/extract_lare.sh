#!/bin/bash

# 0. 환경 설정
export HF_HOME=/data1/huggingface_cache
export CUDA_VISIBLE_DEVICES=0
mkdir -p $HF_HOME

# ----------------------------------------------------------
# 1. Training 데이터 특징 추출
# ----------------------------------------------------------
echo "🚀 [1/3] Extracting features for TRAINING set..."
python extract_lare.py \
  --input_path annotation/my_train_list.txt \
  --output_path /data1/lare_features/train \
  --t 200 \
  --prompt 'a photo' \
  --ensemble_size 4 \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --img_size 256 256 \
  --n-gpus 1

# ----------------------------------------------------------
# 2. Validation 데이터 특징 추출
# ----------------------------------------------------------
echo "🚀 [2/3] Extracting features for VALIDATION set..."
python extract_lare.py \
  --input_path annotation/my_val_list.txt \
  --output_path /data1/lare_features/val \
  --t 200 \
  --prompt 'a photo' \
  --ensemble_size 4 \
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
  --img_size 256 256 \
  --n-gpus 1

# ----------------------------------------------------------
# 3. 지도 파일(ann.txt) 합치기
# ----------------------------------------------------------
echo "🚀 [3/3] Merging annotation files..."
# Train과 Valid의 ann.txt를 하나로 합쳐서 훈련 코드에 넘겨줍니다.
mkdir -p /data1/lare_features/merged
cat /data1/lare_features/train/ann.txt /data1/lare_features/val/ann.txt > /data1/lare_features/merged/all_ann.txt

echo "✅ 모든 과정 완료! 통합 맵 파일: /data1/lare_features/merged/all_ann.txt"