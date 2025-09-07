import os
import argparse
import numpy as np
import tensorflow as tf
import yaml

from utils.constants import RESULT_DIR, SRC_DIR, SEED
from utils.data_handler import get_tsc_train_dataset, preprocess_data
from models.clf_wrapper import ClassifierWrapper
import keras
import datetime
np.random.seed(SEED)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Time Series Backdoor Attack')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use (e.g., 0, 1, or 0,1)')
    parser.add_argument("--dataset_name", type=str, default='iAWE', help="Dataset name")
    parser.add_argument("--clf_name", type=str, help="Classifier name")
    parser.add_argument("--data_type", type=str, help="Train Service Provider (sp) or Attacker (atk) model")
    parser.add_argument('--data_ratio', type=float, default=1.0, help='Ratio of data to use for training')

    # Parse command line arguments first
    args, unknown = parser.parse_known_args()
    training_configs = yaml.safe_load(open(f"{SRC_DIR}/configs/training_tsc.yaml", 'r'))
    # Set visible GPU device
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(f"Using GPU: {gpu}")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Other arguments
    dataset_name = args.dataset_name
    clf_name = args.clf_name
    data_type = args.data_type
    data_ratio = args.data_ratio
    training_config = training_configs[clf_name][dataset_name]
    
    print(f"[+] Parameters: dataset_name={dataset_name}, classifier={clf_name}, data_type={data_type}, data_ratio={data_ratio}")
    print(f"[+] Training configuration: {training_config}")
        
    # Import the dataset
    print(f"Loading dataset: {dataset_name}")
    x_train, y_train, x_test, y_test = get_tsc_train_dataset(
        dataset_name=dataset_name,
        data_ratio=data_ratio,
        data_type=data_type
    )
    print(f"Data shapes: {x_train.shape}, {y_train.shape}, {x_test.shape}, {y_test.shape}")
    
    # Prepare the data
    x_train, y_train, x_test, y_test, enc = preprocess_data(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    # Create the classifier
    today = datetime.datetime.now().strftime("%Y%m%d")
    output_directory = os.path.join(RESULT_DIR, dataset_name, clf_name, data_type, today)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    print(f"[+] Output directory: {output_directory}")
    nb_classes = enc.categories_[0].shape[0]
    print(f"[+] Number of classes: {nb_classes}")
    # Training the classifier
    classifier = ClassifierWrapper(
        training_config=training_config,
        clf_name=clf_name,
        input_shape=x_train.shape[1:],
        nb_classes=nb_classes,
        output_directory=output_directory,
        verbose=True
    )
    classifier.train(x_train, y_train, x_test, y_test)
    print(f"[+] Classifier {clf_name} trained successfully.")