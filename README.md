# Black-box Backdoor Attack on Continual Semi-Supervised Learning time-series-based IoT System

This repository contains the official implementation for the paper: "Black-box Backdoor Attack on Continual Semi-Supervised Learning time-series-based IoT System".

We propose a novel black-box backdoor attack for CSSL-based time-series classification that is gradient-free, annotation-free, and update-efficient. Our attack pipeline consists of three main stages:

(a) Surrogate Model Training: Training a surrogate model via query-based knowledge distillation.

(b) Trigger Generator Training: Optimizing a generator to produce effective backdoor triggers.

(c) Backdoor Training: Injecting poisoned samples to progressively reinforce the backdoor through the CSSL update mechanism.

![Alt text](images/Methodology.png)

# 🔧 Prerequisites

## Hardware

- GPU: All experiments were conducted on an NVIDIA A100 (40GB). A GPU with at least 24GB of VRAM is recommended.
- DRAM: A minimum of 32GB of RAM is required.

## Environment:

We recommend creating a virtual environment to manage dependencies, using conda or pip.

```console
pip install -r blackbox_tsc_env.yml
```

Or just download these packages:

```console
tensorflow>=2.10.
scikit-learn
joblib
pandas
numpy
matplotlib
yaml
seaborn
```

# 📊 Data & Model Weights

You can download the necessary datasets and pre-trained model weights from the links below, `sp`: Refers to the Service Provider's data, `atk`: Refers to the Attacker's data. Please contact us through this google form for the password: https://forms.gle/fRDQQ34SPQ2dAp8Z8.

For the original time-series datasets, please refer to:

- iAWE: https://iawe.github.io/
- MotionSense: https://github.com/mmalekzadeh/motion-sense

- Our preprocessed training & test data can be download here (Numpy format):
  - iAWE train/test dataset: [dataset](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EVmT9-YHRHhChJxX4izF3hcB4n72DMPABAKbSI7DWX-Q9Q?e=yXacW9).
  - MotionSense train/test dataset: [dataset](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EeslKSYLfWBHvLa6lLbFS9cB3KKqqXpfMA1YuWlGOUK4Og?e=dF0heC).

Pre-trained model weights can be download here:

- iAWE pretrained models: [iAWE pretrained models](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EQi0fD0MXLdDsGsagxMQibwBHlKa25dp6XPINiCmK3Zo3Q?e=zdCRXq).
- MotionSense pretrained models: [MotionSense pretrained models](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EUe7Oek1ZHRNr9xUfUbv3BsBAhrqwKNjyQPTByWp32zA6g?e=XvAAq2).

The model weights for the blackbox backdoor attacks can be download here:

- iAWE blackbox backdoor attacks: [pretrained models](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EeJo2SQ7YVdGj5ZzPTeTqKYBqYYFFpqymcQImOMDoVy4_g?e=0VZ8Us).
- MotionSense blackbox backdoor attacks: [pretrained models](https://tngnemo-my.sharepoint.com/:u:/g/personal/levels1912_tngnemo_onmicrosoft_com/EWDSSiEBCvxKigYu2UWACq4BzOvlXnyIBcF1NLuuNEYzyQ?e=Y15tfb).

# 🚀 Running the Experiments

**Please download and place the data and model weights in the appropriate directories before running the scripts.**

## 0. Setup

- Please make sure you have the following dependencies installed:

```console
pip install -r blackbox_tsc_env.yml
```

- Steps to reproduce the results
  - Step 1: Train the initial models for both the attacker (surrogate) and the service provider (target).
  - Step 2: Evaluate the attacks (white-box, black-box, gray-box).
  - Step 3: Update efficiency section, where you need an existed black-box backdoor attack model. This is on `scripts/continuous_bd.sh`.

## 1. Train Initial Models

- First, train the initial time-series classification models for both the attacker (surrogate) and the service provider (target). Or you can download the pretrained versions on the link above.

```console
sh scripts/train_tsc.sh
```

## 2. Execute the Black-Box Backdoor Attack

- This script runs the core attack pipeline, including surrogate model training and trigger generation.

```console
sh scripts/black_box_attacks.sh.
```

- Key arguments:
- `SURROGATE_CLF`: The architecture of the surrogate classifier..
- `TARGET_CLF`: The architecture of the target classifier.
- `EXP_NO`: The experiment number for logging purposes.
- `MAIN_EPOCHS`: Number of target model update cycles (default: 50).
- `ATK_EPOCHS` ($\kappa$): Number of attack rounds (including surrogate model training, and trigger generator training).
- `backdoor_training`: do backdoor training (the (c) step on our method), comment this if you want to run blackbox adversarial attacks.
- Other hyperparameters, stay fixed are in the `configs/*.yaml` files.

## 3. Black-box backdoor attack Attack evaluations

- First, please download the data and models weights from the links above.
- Then, for black-box backdoor attack evaluations on the full dataset, please go to `notebook/run_attacks.ipynb`.

## 4. Gray-box injection attack

- To run the gray-box injection attack, use the following script:

```console
sh scripts/gray_box_attacks.sh
```

## 4. Update efficiency

- This experiment evaluates the persistence and efficiency of the backdoor over multiple updates. It requires a pre-trained trigger generator and a backdoored target model from Step 2.
- iAWE: Assumes the target model is `FCN` and the target class is `computer` (label 2).
- MotionSense: Assumes the target model is `Transformer` and the target class is `sit` (label 2).

To run the experiment, use the following script:

```console
sh scripts/continuous_bd.sh
```

# ⚙️ Hyperparameters and setups

- The optimal hyperparameters vary depending on the target model and dataset. For full transparency and reproducibility, a comprehensive list of all settings used in our experiments is available in the following Google Sheet: ➡️ [Google Sheet](https://docs.google.com/spreadsheets/d/1rab4JrnUre5lL9s6hdRscOIStbhpGMOjvgPmyhnJCPs/edit?usp=sharing).
- A summary of the most critical hyperparameters is provided below:

| **Parameter**                        |   **iAWE**   | **MotionSense** |
| :----------------------------------- | :----------: | :-------------: |
| Query Budget ($\beta_{init}$)        |     2000     |       400       |
| Injection Budget ($\beta_{inject}$)  |     4000     |       800       |
| Maximum Target Model Update Times    |      50      |       50        |
| Attacker's step ($\kappa$)           |      2       |        2        |
| Maximum Amplitude Ratio ($\epsilon$) | `[0.1, 0.6]` |  `[0.1, 0.6]`   |
| Generator Training epoch             |      40      |       50        |
| Generator Learning rate              |   1.00e-03   |    1.00e-03     |
| TSC Update Training Epochs           |      10      |       100       |
| TSC Update Learning rate             |   1.00e-03   |    1.00e-03     |
| Generator data ratio                 |     0.3      |       0.3       |
