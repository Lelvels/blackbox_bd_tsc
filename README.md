# Black-box Backdoor Attack on Continual Semi-Supervised Learning time-series-based IoT System

This is the code is the detailed implementation for the black-box backdoor attacks "Black-box Backdoor Attack on Continual Semi-Supervised Learning time-series-based IoT System".

## Prerequisites
- Please install the environment followed the .env file.

## Data
- The data can be download on [link].

## How to run
- Our Methodology - Blackbox backdoor attacks:
python black_box_attacks.sh

- The steps to run the attacks:

## Hyperparameters and setups
- We configure hyperparameters to match the scale of each dataset. The attacker’s initial query budget ($\beta_{init}$) is 2000 for iAWE and 400 for MotionSense, with subsequent injection budgets ($\beta_{inject}$) of 4000 and 800, respectively. Attacks are simulated over 50 target model update cycles with two inner attack rounds ($\kappa=2$). The trigger generator is trained for up to 50 epochs, with amplitude ratio $\alpha \in [0.1, 0.5]$ depending on the target class. During CSSL updates, the target model is retrained for 10 epochs (iAWE) and 100 epochs (MotionSense). All models, including the surrogate ResNet, are trained with Adam (learning rate $10^{-3}$) and a \textit{ReduceLROnPlateau} scheduler. Experiments are implemented in TensorFlow 2~\cite{abadi2016tensorflow} and run on a single NVIDIA A100 GPU.
