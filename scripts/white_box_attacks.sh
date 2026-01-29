#!/bin/sh

# Set the target class and experiment number
EXP_NO=3
TARGET_CLASS=4
TARGET_CLF=fcn
WHITEBOX_INJECTION=True

# Attack parameters
MAIN_EPOCHS=50
AMPLITUDE=0.32
AMPLITUDE_REG_WEIGHT=1e-3
GRAYBOX_MODE=False

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
                            --gpu ${GPU} \
                            --graybox_mode ${GRAYBOX_MODE}