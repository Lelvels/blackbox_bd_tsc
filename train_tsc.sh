#!/bin/sh
#SBATCH --job-name=ms_tsc_atk
#SBATCH --out="/home/fmg2/v-thanh/Code/results/TSBA/logs/MotionSense_transformer_tsc_atk.log"
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tesla_a100:1
# gpu:tesla_a6000_50g:1 or gpu:tesla_a100:1

source /home/fmg2/v-thanh/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /home/fmg2/v-thanh/Code/source/Time_Series_Backdoor_Attack
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