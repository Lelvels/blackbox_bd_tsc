#!/bin/sh
#SBATCH --job-name="adv_t4_res_res"
#SBATCH --out="/home/fmg2/v-thanh/Code/results/TSBA/logs/adv_t4_res_res.log"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:tesla_a100_80g:1
# tesla_a100_80g / tesla_a6000ada_50g and tesla_a100

# Surrogate and target classifiers with resnet as default surrogate classifier
SURROGATE_CLF=resnet
TARGET_CLF=resnet
    
# Set the target class and experiment number
EXP_NO=0
TARGET_CLASS=4
AMPLITUDE=0.45

# Maximum number of queries for the attack
DATASET=iAWE
FT_BUDGET=5000
INJECT_BUDGET=7500
ATK_DATA_RATIO=1.0
SP_DATA_RATIO=1.0
GEN_TRAIN_DATA_RATIO=0.3
AMPLITUDE_REG_WEIGHT=1e-3

# MotionSense dataset parameters
# DATASET=MotionSense
# FT_BUDGET=400
# INJECT_BUDGET=800
# ATK_DATA_RATIO=1.0
# SP_DATA_RATIO=1.0
# GEN_TRAIN_DATA_RATIO=0.3
# AMPLITUDE_REG_WEIGHT=1e-3

# Attack epochs
MAIN_EPOCHS=50
ATK_EPOCHS=2

# Generate dynamic job name and output path
source /home/fmg2/v-thanh/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /home/fmg2/v-thanh/Code/source/Time_Series_Backdoor_Attack

# Target class and experiment number
export TF_FORCE_GPU_ALLOW_GROWTH=1
export KERAS_BACKEND="tensorflow"
export CUDA_VISIBLE_DEVICES="0"

# Set environment variable to allow GPU memory growth
python black_box_attacks.py --dataset_name ${DATASET} \
                            --exp_name "exp_${EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}-ampl_${AMPLITUDE}" \
                            --surrogate_clf_name ${SURROGATE_CLF} \
                            --target_clf_name ${TARGET_CLF} \
                            --target_class ${TARGET_CLASS} \
                            --ft_loss "kl_loss" \
                            --atk_data_ratio ${ATK_DATA_RATIO} \
                            --sp_data_ratio ${SP_DATA_RATIO} \
                            --generator_train_data_ratio ${GEN_TRAIN_DATA_RATIO} \
                            --main_epochs ${MAIN_EPOCHS} \
                            --atk_epochs ${ATK_EPOCHS} \
                            --inject_budget ${INJECT_BUDGET} \
                            --ft_budget ${FT_BUDGET} \
                            --amplitude ${AMPLITUDE} \
                            --amplitude_reg_weight ${AMPLITUDE_REG_WEIGHT} \
                            --gpu ${CUDA_VISIBLE_DEVICES} \
                            --resume \
                            # --backdoor_training #Uncomment this line for our method, comment this line for the baseline