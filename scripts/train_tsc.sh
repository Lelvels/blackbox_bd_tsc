#!/bin/sh

source <path_to_conda>/conda.sh
conda activate my_env
cd <path_to_conda>/Time_Series_Backdoor_Attack
DATASET=MotionSense
data_ratio=1.0

#FCN
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "resnet" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

# python train_tsc.py --dataset $DATASET \
#                     --clf_name "resnet" \
#                     --data_type "atk" \
#                     --data_ratio $data_ratio

# Resnet
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "resnet" \
#                     --data_type "atk" \
#                     --data_ratio $data_ratio

# python train_tsc.py --dataset $DATASET \
#                     --clf_name "resnet" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

# FCN
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "fcn" \
#                     --data_type "atk" \
#                     --data_ratio $data_ratio

# python train_tsc.py --dataset $DATASET \
#                     --clf_name "fcn" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

# LSTM
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "lstm" \
#                     --data_type "atk" \
#                     --data_ratio $data_ratio

# python train_tsc.py --dataset $DATASET \
#                     --clf_name "lstm" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

# GRU
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "gru" \
#                     --data_type "atk" \
#                     --data_ratio $data_ratio

# python train_tsc.py --dataset $DATASET \
#                     --clf_name "gru" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

# Transformer
# python train_tsc.py --dataset $DATASET \
#                     --clf_name "transformer" \
#                     --data_type "sp" \
#                     --data_ratio $data_ratio

python train_tsc.py --dataset $DATASET \
                    --clf_name "transformer" \
                    --data_type "atk" \
                    --data_ratio $data_ratio