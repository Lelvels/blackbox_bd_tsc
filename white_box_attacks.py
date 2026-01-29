import os
import argparse
import yaml
import numpy as np
import csv
import tensorflow as tf
import keras
from models.clf_wrapper import ClassifierWrapper
from models.trigger_gen import TriggerGenerator

from utils.constants import RESULT_DIR, SRC_DIR, SEED
from utils.utils import limit_gpu_memory
from utils.data_handler import get_tsc_train_dataset, preprocess_data, create_dataset, select_stratified_by_ratio
import logging

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(SEED)
limit_gpu_memory()

def setup_logger(log_file=None):
    logger = logging.getLogger('white_box_attacks')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def attack_white_box(output_directory: str,
                    trigger_gen: TriggerGenerator,
                    target_clf: ClassifierWrapper,
                    gen_data_ratio: float,
                    x_train_atk, y_train_atk,
                    x_test_atk, y_test_atk,
                    x_train_sp, y_train_sp,
                    x_test_sp, y_test_sp,
                    y_target,
                    bd_training_config: dict,
                    graybox_mode: bool,
                    logger: logging.Logger,
                    amplitude_reg_weight: float = 2e-3,
                    main_epochs=50):
    """
    White-box backdoor attack implementation.
    
    Args:
        trigger_gen: TriggerGenerator instance
        x_train, y_train: Training data
        x_test, y_test: Test data
        y_target: Target class for backdoor
        logger: Logger instance
        main_epochs: Number of main training epochs
        graybox_mode: Graybox mode, with access to the model's weights, but cannot manually set the labels for the triggered samples.
    """
    # Check the clean model accuracy
    logger.info(f"[+] Starting white-box training with target class {y_target}")
    clean_accuracy = target_clf.model.evaluate(x_test_sp, y_test_sp, verbose=0)[1]
    logger.info(f"[+] Target model clean accuracy: {clean_accuracy:.4f}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    csv_file = os.path.join(output_directory, 'noise_gan_backdoor_training_log.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', "Clean Accuracy", "Train ASR", "Test ASR"])

    # Main training loop parameters
    generator_epochs = trigger_gen.generator_config["generator_epochs"]
    batch_size = trigger_gen.generator_config["batch_size"]
    learning_rate = trigger_gen.generator_config["learning_rate"]

    # Logging configuration
    logger.info('=' * 60)
    logger.info('[CONFIGURATION] Starting Backdoor Attack')
    logger.info('=' * 60)
    
    # Attack mode
    if graybox_mode:
        logger.info(f"[MODE] Gray-box attack to target class {y_target}")
    else:
        logger.info(f"[MODE] TSBA baseline - manually setting triggered samples to target class {y_target}")
    
    # Dataset information
    logger.info(f"[DATA] ATK samples - Train: {x_train_atk.shape[0]}, Test: {x_test_atk.shape[0]}")
    logger.info(f"[DATA] SP samples - Train: {x_train_sp.shape[0]}, Test: {x_test_sp.shape[0]}")
    
    # Attack parameters
    logger.info(f"[PARAMS] Target class: {y_target}")
    logger.info(f"[PARAMS] Budget limit: {gen_data_ratio * 100:.2f}% of total queries")
    logger.info(f"[PARAMS] Main epochs: {main_epochs}")
    logger.info(f"[PARAMS] Amplitude regularization weight: {amplitude_reg_weight}")
    
    # Generator configuration
    logger.info(f"[GENERATOR] Type: {trigger_gen.gen_type}")
    logger.info(f"[GENERATOR] Pattern amplitude: {trigger_gen.max_amplitude}")
    logger.info(f"[GENERATOR] Training epochs: {generator_epochs}")
    logger.info(f"[GENERATOR] Batch size: {batch_size}")
    
    # Training configurations
    logger.info(f"[CONFIG] Backdoor training: {bd_training_config}")
    
    logger.info('=' * 60)

    # Save the initial weights of the generator
    try:
        trigger_gen.output_directory = output_directory
        trigger_gen.generator.save(os.path.join(trigger_gen.output_directory, 'generator_initial.keras'))
        logger.info("[SAVE] Initial generator model saved successfully")
    except Exception as e:
        logger.warning(f"[SAVE] Could not save initial generator weights: {e}")

    # Begin training
    logger.info('\n' + '=' * 60)
    logger.info('[TRAINING] Starting Noise-GAN Backdoor Training')
    logger.info('=' * 60)

    # Create the target labels
    y_train_target = trigger_gen.enc.transform(np.array([y_target]*len(x_train_atk)).reshape(-1, 1)).toarray()
    y_test_target = trigger_gen.enc.transform(np.array([y_target]*len(x_test_atk)).reshape(-1, 1)).toarray() 
    
    # Start the main training loop
    for e in range(1, main_epochs+1):
        logger.info(f"[+] Main Epoch: {e}/{main_epochs}")
        
        # Set the output directory for this epoch
        epoch_output_dir = os.path.join(main_out_dir, f'epoch_{e}')
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        # Train noise generator using custom training function
        logger.info('=' * 20 + ' Training noise generator ' + '=' * 20)
        logger.info("[+] Selecting stratified subsets for generator training and testing...")
        x_train_gen, y_train_gen = select_stratified_by_ratio(
            x_train_atk, y_train_atk,
            data_ratio=gen_data_ratio,
            random_state=SEED + e
        )
        x_test_gen, y_test_gen = select_stratified_by_ratio(
            x_test_atk, y_test_atk,
            data_ratio=gen_data_ratio,
            random_state=SEED + e
        )
        logger.info(f"[+] Selected samples for generator - Train: {x_train_gen.shape[0]}, Test: {x_test_gen.shape[0]}")

        # Create target labels based on actual selected sample sizes
        actual_train_samples = len(x_train_gen)
        actual_test_samples = len(x_test_gen)
        logger.info(f"[+] Actually selected samples - Train: {actual_train_samples}, Test: {actual_test_samples}")

        # Create target labels with correct dimensions
        y_train_gen = trigger_gen.enc.transform(np.array([y_target]*actual_train_samples).reshape(-1, 1)).toarray()
        y_test_gen = trigger_gen.enc.transform(np.array([y_target]*actual_test_samples).reshape(-1, 1)).toarray()

        # Verify dimensions match
        assert len(x_train_gen) == len(y_train_gen), f"Train dimensions mismatch: {len(x_train_gen)} vs {len(y_train_gen)}"
        assert len(x_test_gen) == len(y_test_gen), f"Test dimensions mismatch: {len(x_test_gen)} vs {len(y_test_gen)}"

        # Create training and validation datasets for generator
        gen_train_dataset = create_dataset(x_train_gen, y_train_gen,
                                        batch_size=batch_size,
                                        shuffle=True, prefetch=True).cache()
        gen_val_dataset = create_dataset(x_test_gen, y_test_gen,
                                        batch_size=batch_size,
                                        shuffle=False, prefetch=True).cache()
        clean_test_dataset = create_dataset(x_test_sp, y_test_sp,
                                            batch_size=batch_size,
                                            shuffle=False, prefetch=True).cache()
        
        # Use the custom train_generator function
        trigger_gen.generator.trainable = True
        target_clf.model.trainable = False
        trigger_gen.output_directory = os.path.join(epoch_output_dir, 'generator')
        trigger_gen.train_generator_full_model(
            train_dataset=gen_train_dataset,
            val_dataset=gen_val_dataset,
            surro_model_wrapper=target_clf,
            logger=logger,
            epochs=generator_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            amplitude_reg_weight=amplitude_reg_weight
        )

        # Apply the trigger to the training and test data
        train_triggered_dataset = create_dataset(trigger_gen.apply_trigger(x_train_atk), y_train_target,
                                               batch_size=batch_size,
                                               shuffle=True, prefetch=True).cache()
        test_triggered_dataset = create_dataset(trigger_gen.apply_trigger(x_test_atk), y_test_target,
                                                batch_size=batch_size,
                                                shuffle=False, prefetch=True).cache()

        # Evaluate after training the generator
        bd_ca = target_clf.model.evaluate(clean_test_dataset, verbose=0)[1]
        attack_train_evaluation = target_clf.model.evaluate(train_triggered_dataset, verbose=0)
        attack_test_evaluation = target_clf.model.evaluate(test_triggered_dataset, verbose=0)   
        bd_train_loss, bd_train_asr = attack_train_evaluation[0], attack_train_evaluation[1]
        bd_test_loss, bd_test_asr = attack_test_evaluation[0], attack_test_evaluation[1]
        logger.info(f"Clean accuracy (test): {bd_ca:.4f}")
        logger.info(f"Attack Success Rate (train): {bd_train_asr:.4f} - Loss: {bd_train_loss:.4f}")
        logger.info(f"Attack Success Rate (test): {bd_test_asr:.4f} - Loss: {bd_test_loss:.4f}")

        # Setting trainable flags for training
        target_clf.model.trainable = True
        trigger_gen.generator.trainable = False

        # Collect data for backdoor training
        logger.info(f"[+] Collecting data for backdoor training...")
        selected_x_train_bd = np.concatenate((x_train_gen, x_test_gen), axis=0)
        # Gray-box scenario: You cannot manually set the labels for the triggered samples, 
        # and the updates on the target model will use the predicted labels from the target model.
        # but you have access to the model's weights.
        if graybox_mode:
            # Apply trigger once and reuse
            x_train_for_bd = trigger_gen.apply_trigger(selected_x_train_bd)
            
            # Get predictions in batches to avoid memory issues
            y_predicted_backdoor = []
            for i in range(0, len(x_train_for_bd), batch_size):
                batch_end = min(i + batch_size, len(x_train_for_bd))
                batch_pred = target_clf.model.predict(
                    x_train_for_bd[i:batch_end], 
                    verbose=0
                )
                y_predicted_backdoor.append(batch_pred)
            y_predicted_backdoor = np.concatenate(y_predicted_backdoor, axis=0)
            
            # More efficient one-hot conversion
            y_clf_prediction = np.eye(y_predicted_backdoor.shape[1])[np.argmax(y_predicted_backdoor, axis=1)]
            
            # Convert to class labels for logging
            y_pred_classes = trigger_gen.enc.inverse_transform(y_clf_prediction)
            logger.info(f"[+] Classifier Predicted Class distribution: {np.unique(y_pred_classes, return_counts=True)}")
            
            # Set variables for training
            y_train_for_bd = y_clf_prediction
        else:
            # White-box scenario: You have access to the model's weights 
            # and can manually set the labels for the triggered samples.
            x_train_for_bd = trigger_gen.apply_trigger(selected_x_train_bd)
            target_labels = np.full(len(x_train_for_bd), y_target)
            y_train_for_bd = trigger_gen.enc.transform(target_labels.reshape(-1, 1)).toarray()

        # Combine the poisoned data with the clean data during backdoor training
        logger.info(f"[+] Number of samples for backdoor training (x, y): {x_train_for_bd.shape[0]}, {y_train_for_bd.shape[0]}")
        x_train_bd_combined = np.concatenate((x_train_for_bd, x_train_sp), axis=0)
        y_train_bd_combined = np.concatenate((y_train_for_bd, y_train_sp), axis=0)

        # Train backdoor classifier
        logger.info('=' * 20 + " Training combined backdoor classifier " + '=' * 20)
        target_clf.output_directory = os.path.join(epoch_output_dir, 'backdoor_classifier')
        target_clf.train(
            x_train=x_train_bd_combined,
            y_train=y_train_bd_combined,
            x_val=x_test_sp,
            y_val=y_test_sp,
            epochs=bd_training_config['epochs'],
            batch_size=bd_training_config['batch_size']
        )
        
        # Append results to CSV
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([e, f"{bd_ca:.4f}", f"{bd_train_asr:.4f}", f"{bd_test_asr:.4f}"])

        # Save the generator model after each epoch (using weights to avoid Lambda layer issues)
        generator_model_dir = os.path.join(trigger_gen.output_directory, 'generator_save')
        if not os.path.exists(generator_model_dir):
            os.makedirs(generator_model_dir, exist_ok=True)
        
        try:
            # Save weights (safer approach)
            generator_weights_path = os.path.join(generator_model_dir, f'generator_epoch_{e}.keras')
            trigger_gen.generator.save(generator_weights_path)
            logger.info(f"[+] Generator weights saved to {generator_weights_path}")
        except Exception as e:
            logger.warning(f"[+] Could not save generator weights: {e}")

        # Add space between epochs
        logger.info('\n' * 2)

    # End of training
    logger.info(f"[+] Training completed. Models saved to {trigger_gen.output_directory}")

if __name__ == "__main__":
    # System and GPU configuration
    parser = argparse.ArgumentParser(description='Time Series Backdoor Attack - White Box')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device ID to use (e.g., 0, 1, or 0,1)')
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    
    # Dataset and experiment configuration
    parser.add_argument("--dataset_name", type=str, default='MotionSense', help="Dataset name.")
    parser.add_argument("--exp_name", type=str, default='', help="Experiment name.")

    # Data ratios for attacker and service provider
    parser.add_argument("--atk_data_ratio", type=float, default=1.0, help="Ratio of data to use for the attack (default: 1.0)")
    parser.add_argument("--sp_data_ratio", type=float, default=1.0, help="Ratio of data to use for the service provider (default: 1.0)")
    parser.add_argument("--gen_data_ratio", type=float, default=0.3, help="Ratio of data to use for the generator (default: 0.3)")

    # Classifier parameters
    parser.add_argument("--target_clf", type=str, default='fcn', help="Target classifier name")
    parser.add_argument("--clf_dir", type=str, default='', help="Directory to the pre-trained target classifier")
    
    # Attack configuration
    parser.add_argument("--target_class", type=int, default=0, help="Target class for backdoor attack.")
    parser.add_argument("--main_epochs", type=int, default=50, help="Number of training epochs, each epochs including training generator, and backdoor model update.")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Amplitude for the dynamic amplitude CNN generator (default: 0.2)")
    parser.add_argument("--amplitude_reg_weight", type=float, default=2e-3, help="Amplitude regularization weight for the generator (default: 2e-3)")

    # Attack configuration
    parser.add_argument("--graybox_mode", type=lambda x: x.lower() in ("true", "t", "yes", "y", "1"), default=True, help="Gray-box mode with access to the model's weights, but cannot manually set the labels for the triggered samples.")
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Set visible GPU device
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(f"Using GPU: {gpu}")

    # Extract arguments
    dataset_name = args.dataset_name
    exp_name = args.exp_name

    # Total queries and data ratios
    atk_data_ratio = args.atk_data_ratio
    sp_data_ratio = args.sp_data_ratio
    gen_data_ratio = args.gen_data_ratio

    # Target classifier and directory
    target_clf_name = args.target_clf
    clf_dir = args.clf_dir

    # Target class and training parameters
    y_target = args.target_class
    main_epochs = args.main_epochs
    amplitude = args.amplitude
    amplitude_reg_weight = args.amplitude_reg_weight

    # White-box injection flag
    graybox_mode = args.graybox_mode
    
    # Load configurations
    clf_configs = yaml.safe_load(open(f"{SRC_DIR}/configs/model_update.yaml", 'r'))
    generator_configs = yaml.safe_load(open(f"{SRC_DIR}/configs/training_generator.yaml", 'r'))
    clf_config = clf_configs[target_clf_name][dataset_name]
    generator_config = generator_configs["whitebox"][target_clf_name][dataset_name]
    generator_config['amplitude'] = amplitude  # Override amplitude

    # Set up output directory and logger
    if graybox_mode:
        main_out_dir = os.path.join(RESULT_DIR, dataset_name, "graybox_mode")
    else:
        main_out_dir = os.path.join(RESULT_DIR, dataset_name, "whitebox_tsba")
    
    # If experiment name is provided, use it
    if exp_name == '':
        main_out_dir = os.path.join(main_out_dir, f"wb_clf_{target_clf_name}_target_{y_target}")
    else:
        main_out_dir = os.path.join(main_out_dir, f"wb_{exp_name}")
    
    if not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)
    log_file = os.path.join(main_out_dir, f"wb_{target_clf_name}.log")
    logger = setup_logger(log_file)

    # Log configuration
    logger.info("=" * 80)
    logger.info("[CONFIGURATION] White-box Generator Training Setup")
    logger.info("=" * 80)
    
    # System Configuration
    logger.info(f"[SYSTEM] GPU: {gpu}")
    logger.info(f"[SYSTEM] Output Directory: {main_out_dir}")
    logger.info(f"[SYSTEM] Seed: {args.seed}")
    
    # Dataset Configuration
    logger.info(f"[DATASET] Name: {dataset_name}")
    logger.info(f"[DATASET] Attacker Data Ratio: {atk_data_ratio}")
    logger.info(f"[DATASET] Service Provider Data Ratio: {sp_data_ratio}")
    logger.info(f"[DATASET] Target Class: {y_target}")
    
    # Model Configuration
    logger.info(f"[MODEL] Target Classifier: {target_clf_name}")
    logger.info(f"[MODEL] Classifier Directory: {clf_dir if clf_dir else 'Default'}")
    
    # Training Configuration
    logger.info(f"[TRAINING] Main Epochs: {main_epochs}")
    logger.info(f"[GENERATOR] Amplitude: {amplitude}")
    logger.info(f"[GENERATOR] Amplitude Regularization Weight: {amplitude_reg_weight}")

    # Training Configuration
    logger.info(f"[TRAINING CONFIG] Classifier Config: {clf_config}")
    logger.info(f"[TRAINING CONFIG] Generator Config: {generator_config}")
    
    # Attack Configuration
    logger.info(f"[ATTACK] Mode: {'Graybox Mode' if graybox_mode else 'TSBA Whitebox Baseline'}")
    logger.info(f"[ATTACK] Experiment Name: {exp_name if exp_name else 'Default'}")
    
    logger.info("=" * 80)

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    x_train_atk, y_train_atk, x_test_atk, y_test_atk = get_tsc_train_dataset(
        dataset_name=dataset_name,
        data_ratio=atk_data_ratio,
        data_type="atk"
    )
    logger.info(f"[+] Attacker data shapes: {x_train_atk.shape}, {y_train_atk.shape}, {x_test_atk.shape}, {y_test_atk.shape}")

    # Preprocess data
    x_train_atk, y_train_atk, x_test_atk, y_test_atk, enc = preprocess_data(
        x_train=x_train_atk,
        y_train=y_train_atk,
        x_test=x_test_atk,
        y_test=y_test_atk,
    )

    # Load the surrogate classifier dataset
    x_train_sp, y_train_sp, x_test_sp, y_test_sp = get_tsc_train_dataset(
        dataset_name=dataset_name,
        data_ratio=sp_data_ratio,
        data_type="sp"
    )
    logger.info(f"Service provider data shapes: {x_train_sp.shape}, {y_train_sp.shape}, {x_test_sp.shape}, {y_test_sp.shape}")

    # Preprocess data
    x_train_sp, y_train_sp, x_test_sp, y_test_sp, enc = preprocess_data(
        x_train=x_train_sp,
        y_train=y_train_sp,
        x_test=x_test_sp,
        y_test=y_test_sp
    )

    # Log dataset information
    input_shape = x_train_atk.shape[1:]
    nb_classes = enc.categories_[0].shape[0]
    logger.info(f"[+] Input shape: {input_shape}, Number of classes: {nb_classes}")

    # Load pre-trained classifier
    classifier = ClassifierWrapper(
        training_config=clf_config,
        clf_name=target_clf_name,
        input_shape=input_shape,
        nb_classes=nb_classes,
        output_directory=None,
        verbose=True
    )
    
    # Load model weights & Set up classifier directory
    if not os.path.exists(clf_dir):
        target_clf_dir = os.path.join(RESULT_DIR, dataset_name, target_clf_name, "sp")
    else:
        target_clf_dir = clf_dir
    # Load pre-trained classifier
    logger.info(f"Loading pre-trained classifier from {target_clf_dir}")
    clf_path = os.path.join(target_clf_dir, "best_model.keras")
    if os.path.exists(clf_path):
        classifier.model = keras.models.load_model(clf_path)
        logger.info(f"Loaded full model from {clf_path}")
    else:
        weights_path = os.path.join(target_clf_dir, "best_model.keras")
        if os.path.exists(weights_path):
            classifier.model.load_weights(weights_path)
            logger.info(f"Loaded weights from {weights_path}")
        else:
            raise FileNotFoundError(f"No model found at {target_clf_dir}")

    # Initialize trigger generator
    trigger_gen = TriggerGenerator(
        output_directory=None,
        generator_config=generator_config,
        input_shape=input_shape,
        max_amplitude=amplitude,
        gen_type='dynamic_ampl_cnn',
        enc=enc
    )
    
    # Start white-box attack
    logger.info("Starting white-box backdoor attack...")
    attack_white_box(
        output_directory=main_out_dir,
        trigger_gen=trigger_gen,
        target_clf=classifier,
        gen_data_ratio=gen_data_ratio,
        x_train_atk=x_train_atk,
        x_test_atk=x_test_atk,
        y_train_atk=y_train_atk,
        y_test_atk=y_test_atk,
        x_train_sp=x_train_sp,
        y_train_sp=y_train_sp,
        x_test_sp=x_test_sp,
        y_test_sp=y_test_sp,
        y_target=y_target,
        bd_training_config=clf_config,
        logger=logger,
        amplitude_reg_weight=amplitude_reg_weight,
        main_epochs=main_epochs,
        graybox_mode=graybox_mode
    )

    # End of attack
    logger.info(f"White-box attack completed. Results saved in {main_out_dir}")