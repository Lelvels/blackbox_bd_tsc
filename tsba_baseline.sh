#!/bin/sh
#SBATCH --job-name=ms0-trans
#SBATCH --out="/home/fmg2/v-thanh/Code/results/TSBA/logs/ms0-trans.log"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tesla_a100:1

# Set the target class and experiment number
EXP_NO=0
TARGET_CLASS=0
TARGET_CLF=transformer
WHITEBOX_INJECTION=False

# Attack parameters
MAIN_EPOCHS=50
AMPLITUDE=0.4
AMPLITUDE_REG_WEIGHT=1e-3

# iAWE dataset parameters
# DATASET="iAWE"
# ATK_DATA_RATIO=1.0
# SP_DATA_RATIO=1.0
# GEN_DATA_RATIO=0.3
# AMPLITUDE_REG_WEIGHT=1e-3

# MotionSense dataset parameters
DATASET=MotionSense
ATK_DATA_RATIO=1.0
SP_DATA_RATIO=1.0
GEN_DATA_RATIO=0.3
AMPLITUDE_REG_WEIGHT=1e-3

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
                            --exp_name "tsba_exp_${EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}" \
                            --whitebox_injection ${WHITEBOX_INJECTION} \
                            --atk_data_ratio ${ATK_DATA_RATIO} \
                            --sp_data_ratio ${SP_DATA_RATIO} \
                            --main_epochs ${MAIN_EPOCHS} \
                            --amplitude ${AMPLITUDE} \
                            --amplitude_reg_weight ${AMPLITUDE_REG_WEIGHT} \
                            --gpu ${GPU}