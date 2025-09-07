#!/bin/sh
#SBATCH --job-name=ia3-4-fcn
#SBATCH --out="/home/fmg2/v-thanh/Code/results/TSBA/logs/wij_ia3_4_fcn.log"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:tesla_a100_80g:1
# tesla_a100_80g / tesla_a6000ada_50g and tesla_a100

# Set the target class and experiment number
EXP_NO=3
TARGET_CLASS=4
TARGET_CLF=fcn
WHITEBOX_INJECTION=True

# Attack parameters
MAIN_EPOCHS=50
AMPLITUDE=0.32
AMPLITUDE_REG_WEIGHT=1e-3

# iAWE dataset parameters
# DATASET="iAWE"
# ATK_DATA_RATIO=1.0
# SP_DATA_RATIO=1.0
# GEN_DATA_RATIO=0.3

# MotionSense dataset parameters
DATASET=MotionSense
ATK_DATA_RATIO=1.0
SP_DATA_RATIO=1.0
GEN_DATA_RATIO=0.3

# Set environment variable to allow GPU memory growth
export GPU="0"
export TF_FORCE_GPU_ALLOW_GROWTH=1
export KERAS_BACKEND="tensorflow"
export CUDA_VISIBLE_DEVICES="0"

# Generate dynamic job name and output path
source /home/fmg2/v-thanh/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /home/fmg2/v-thanh/Code/source/Time_Series_Backdoor_Attack
python white_box_attacks.py --dataset ${DATASET} \
                            --target_clf ${TARGET_CLF} \
                            --target_class ${TARGET_CLASS} \
                            --exp_name "exp_${EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}" \
                            --whitebox_injection ${WHITEBOX_INJECTION} \
                            --atk_data_ratio ${ATK_DATA_RATIO} \
                            --sp_data_ratio ${SP_DATA_RATIO} \
                            --main_epochs ${MAIN_EPOCHS} \
                            --amplitude ${AMPLITUDE} \
                            --amplitude_reg_weight ${AMPLITUDE_REG_WEIGHT} \
                            --gpu ${GPU}