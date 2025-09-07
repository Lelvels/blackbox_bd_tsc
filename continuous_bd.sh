#!/bin/sh
#SBATCH --job-name="cb4_ia2_res_fcn"
#SBATCH --out="/home/fmg2/v-thanh/Code/results/TSBA/logs/cb4_ia2_res_fcn.log"
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:tesla_a100_80G:1

# Parameters for continuous backdoor attack
EXP_NO=4
ATK_UPDATE_INTERVAL=0

# Attack checkpoint parameters
SURRO_CLF=resnet
TARGET_CLF=fcn
TARGET_CLASS=2
AMPLITUDE=0.4
ORI_EXP_NO=3
GEN_EPOCH=6
GEN_ATK_EPOCH=2
BACKDOOR_EPOCH=$((GEN_EPOCH - 1))

# Maximum number of queries for the attack
DATASET=iAWE
INJECT_BUDGET=5000
FT_BUDGET=7500
ATK_DATA_RATIO=1.0 
SP_DATA_RATIO=1.0
GEN_DATA_RATIO=0.3
AMPLITUDE_REG_WEIGHT=2e-3

# MotionSense dataset parameters
# DATASET=MotionSense
# INJECT_BUDGET=800
# FT_BUDGET=400
# ATK_DATA_RATIO=1.0
# SP_DATA_RATIO=1.0
# GEN_DATA_RATIO=0.3
# AMPLITUDE_REG_WEIGHT=2e-3

# Set the generator and target classifier directories based on your file structure
EPOCHS=50
ATK_EPOCHS=2
GEN_NAME="dynamic_ampl_cnn"
RESULT_DIR="/home/fmg2/v-thanh/Code/results/TSBA"
TARGET_CLF_DIR="${RESULT_DIR}/${DATASET}/with_bd/bb_${SURRO_CLF}_exp_${ORI_EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}/epoch_${BACKDOOR_EPOCH}/target_model_update/best_model.keras"
SURRO_CLF_DIR="${RESULT_DIR}/${DATASET}/with_bd/bb_${SURRO_CLF}_exp_${ORI_EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}/epoch_${BACKDOOR_EPOCH}/surrogate_model_update/best_model.keras"
GEN_DIR="${RESULT_DIR}/${DATASET}/with_bd/bb_${SURRO_CLF}_exp_${ORI_EXP_NO}-t_${TARGET_CLASS}-clf_${TARGET_CLF}/epoch_${GEN_EPOCH}/generator_epoch_${GEN_ATK_EPOCH}/best_generator.keras"

# Debug: Print paths
echo "Target classifier path: $TARGET_CLF_DIR"
echo "Surrogate classifier path: $SURRO_CLF_DIR"
echo "Generator path: $GEN_DIR"

# Check if files exist
if [ ! -f "$TARGET_CLF_DIR" ]; then
    echo "Error: Target classifier not found at: $TARGET_CLF_DIR"
    exit 1
fi

if [ ! -f "$GEN_DIR" ]; then
    echo "Error: Generator not found at: $GEN_DIR"
    exit 1
fi

# Generate dynamic job name and output path
source /home/fmg2/v-thanh/miniconda3/etc/profile.d/conda.sh
conda activate my_env
cd /home/fmg2/v-thanh/Code/source/Time_Series_Backdoor_Attack

# Target class and experiment number
export TF_FORCE_GPU_ALLOW_GROWTH=1
export KERAS_BACKEND="tensorflow"
export CUDA_VISIBLE_DEVICES="0"

# Set environment variable to allow GPU memory growth
python continuous_bd.py --gpu ${CUDA_VISIBLE_DEVICES} \
                        --dataset_name ${DATASET} \
                        --exp_name "exp${EXP_NO}_t${TARGET_CLASS}_surro_${SURRO_CLF}_target_${TARGET_CLF}" \
                        --amplitude ${AMPLITUDE} \
                        --atk_epochs ${ATK_EPOCHS} \
                        --atk_data_ratio ${ATK_DATA_RATIO} \
                        --sp_data_ratio ${SP_DATA_RATIO} \
                        --gen_data_ratio ${GEN_DATA_RATIO} \
                        --target_clf_name ${TARGET_CLF} \
                        --target_clf_dir ${TARGET_CLF_DIR} \
                        --surro_clf_name ${SURRO_CLF} \
                        --surro_clf_dir ${SURRO_CLF_DIR} \
                        --gen_name ${GEN_NAME} \
                        --gen_dir ${GEN_DIR} \
                        --atk_update_interval ${ATK_UPDATE_INTERVAL} \
                        --finetune_budget ${FT_BUDGET} \
                        --injection_budget ${INJECT_BUDGET} \
                        --epochs ${EPOCHS} \
                        --target_class ${TARGET_CLASS} \
                        --amplitude_reg_weight ${AMPLITUDE_REG_WEIGHT} \
                        # --do_injection