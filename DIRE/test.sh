#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire

EXP_NAME="my_custom_experiment"
CKPT="/home/deepfake/lju_workspace/DIRE/checkpoints/lsun_adm.pth"
DATASETS_TEST="/home/deepfake/lju_workspace/DIRE/data_sample/data_re"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST