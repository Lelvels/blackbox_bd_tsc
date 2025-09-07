# Import necessary libraries
import os
import argparse
import yaml
import numpy as np
import logging
import keras
import csv
import tensorflow as tf
from utils.constants import RESULT_DIR, SRC_DIR, SEED
from utils.utils import SimpleBudgetCounter, initialize_csv_logger, log_to_csv
from sklearn.model_selection import train_test_split
from utils.data_handler import get_tsc_train_dataset, preprocess_data, create_dataset, select_stratified_by_ratio, select_stratified_by_sample
from models.clf_wrapper import ClassifierWrapper
from utils.data_storage import DataStorage
from models.trigger_gen import TriggerGenerator
from models.finetuner import SurrogateModelFinetuner

# For reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

def setup_logger(log_file=None):
    logger = logging.getLogger('continuous_bd_logger')
    logger.setLevel(logging.INFO)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous Backdoor Attack on Time Series Classification")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")

    # Dataset and experiment configuration
    parser.add_argument("--dataset_name", type=str, help="Dataset name.")
    parser.add_argument("--exp_name", type=str, help="Experiment name.")
    parser.add_argument("--amplitude", type=float, help="Amplitude for the dynamic amplitude CNN generator")
    parser.add_argument("--atk_epochs", type=int, default=2, help="Number of epochs for the attack generator training (default: 2)")

    # Data configuration
    parser.add_argument('--atk_data_ratio', type=float, default=1.0, help='Ratio of attack data to use')
    parser.add_argument('--sp_data_ratio', type=float, default=1.0, help='Ratio of support data to use')
    parser.add_argument('--gen_data_ratio', type=float, default=0.3, help='Ratio of generator data to use')

    # Classifier parameters and generator configuration
    parser.add_argument("--target_clf_name", type=str, help="Target classifier name")
    parser.add_argument("--target_clf_dir", type=str, help="Directory to the pre-trained target classifier")
    parser.add_argument("--surro_clf_name", type=str, default='resnet', help="Surro classifier name (default: resnet)")
    parser.add_argument("--surro_clf_dir", type=str, help="Directory to the pre-trained Surro classifier")
    parser.add_argument("--gen_name", type=str, help="Path to the generator configuration name")
    parser.add_argument("--gen_dir", type=str, help="Trigger generator directory")

    # Attacks configuration
    parser.add_argument("--atk_update_interval", type=int, default=0, help="Interval for updating the generator (in epochs)")
    parser.add_argument("--do_injection", action='store_true', help='If set, will train the backdoor attack with the surrogate model')
    parser.add_argument("--finetune_budget", type=int, default=400, help="Budget for fine-tuning the target classifier (in samples)")
    parser.add_argument("--injection_budget", type=int, default=800, help="Budget for data injection (in samples)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the target model")
    parser.add_argument("--ft_loss", type=str, default="kl_loss", help="Fine-tune loss")

    # Attack parameters
    parser.add_argument("--target_class", type=int, help="Target class for the backdoor attack")
    parser.add_argument("--amplitude_reg_weight", type=float, default=2e-3, help="Amplitude regularization weight for the generator (default: 2e-3)")
    args = parser.parse_args()

    # Set visible GPU device
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(f"Using GPU: {gpu}")

    # Extract arguments
    dataset_name = args.dataset_name
    exp_name = args.exp_name
    amplitude = args.amplitude
    atk_epochs = args.atk_epochs

    # Attack parameters
    atk_data_ratio = args.atk_data_ratio
    sp_data_ratio = args.sp_data_ratio
    gen_data_ratio = args.gen_data_ratio

    # Target classifier and generator parameters
    target_clf_name = args.target_clf_name
    target_clf_dir = args.target_clf_dir
    surro_clf_name = args.surro_clf_name
    surro_clf_dir = args.surro_clf_dir
    gen_name = args.gen_name
    gen_dir = args.gen_dir

    # Settings
    atk_update_interval = args.atk_update_interval
    do_injection = args.do_injection
    finetune_budget = args.finetune_budget
    inject_budget = args.injection_budget
    main_epochs = args.epochs
    ft_loss = args.ft_loss

    # Attack configuration
    y_target = args.target_class
    amplitude_reg_weight = args.amplitude_reg_weight
    if y_target < 0:
        raise ValueError("Target class must be a non-negative integer.")

    # Load configuration files
    training_configs_path = os.path.join(SRC_DIR, "configs", "model_update.yaml")
    finetune_configs_path = os.path.join(SRC_DIR, "configs", "finetune_surrogate.yaml")
    generator_configs_path = os.path.join(SRC_DIR, "configs", "training_generator.yaml")
    
    training_configs = yaml.safe_load(open(training_configs_path, 'r'))
    finetune_configs = yaml.safe_load(open(finetune_configs_path, 'r'))
    generator_configs = yaml.safe_load(open(generator_configs_path, 'r'))

    # Validate dataset name
    target_clf_config = training_configs[target_clf_name][dataset_name]
    surro_clf_config = training_configs[surro_clf_name][dataset_name]
    finetune_params = finetune_configs[surro_clf_name][dataset_name]
    generator_config = generator_configs["blackbox"][target_clf_name][dataset_name]

    # Experiment name setup
    main_out_dir = os.path.join(RESULT_DIR, dataset_name, "continuous_bd")
    if exp_name == '':
        main_out_dir = os.path.join(main_out_dir, f"{surro_clf_name}_{target_clf_name}_amplitude_{amplitude}")
    else:
        main_out_dir = os.path.join(main_out_dir, f"cb_{exp_name}")

    if not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)

    # Logger setup
    log_file = os.path.join(main_out_dir, "continuous_bd_logs.log")
    logger = setup_logger(log_file)
    logger.info(f"Logging to file: {log_file}")

    # Log configuration
    logger.info("="*80)
    logger.info("CONTINUOUS BACKDOOR ATTACK CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Attacks data ratio: {atk_data_ratio}")
    logger.info(f"Support data ratio: {sp_data_ratio}")
    logger.info(f"Target classifier name: {target_clf_name}")
    logger.info(f"Target classifier directory: {target_clf_dir}")
    logger.info(f"Trigger type: {gen_name}")
    logger.info(f"Trigger directory: {gen_dir}")
    logger.info(f"Amplitude: {amplitude}")
    logger.info(f"Attacker's update interval: {atk_update_interval}")
    logger.info(f"Amplitude regularization weight: {amplitude_reg_weight}")
    logger.info(f"Output directory: {main_out_dir}")
    if do_injection:
        print("[Data Injection] Data injection is enabled.")
    else:
        print("[Data Injection] Data injection is disabled. The attack will not inject poisoned data into the target model.")
    logger.info("="*80)

    # CSV Logger setup
    csv_headers = [
        'Epoch',
        'Surro ASR', 
        'Target ASR',
        'Surro Clean Acc',
        'Target Clean Acc',
        'Surro ASR Loss',
        'Target ASR Loss',
        'FT Train Loss',
        'FT Val Loss',
        'Update Samples Count'
    ]
    csv_file_path = os.path.join(main_out_dir, f"continuous_attack_metrics_{target_clf_name}.csv")
    initialize_csv_logger(csv_file_path, csv_headers)
    logger.info(f"[CSV Logger] Metrics will be saved to: {csv_file_path}")

    # Load dataset
    logger.info(f"Loading dataset {dataset_name}...")
    x_train_atk, y_train_atk, x_test_atk, y_test_atk = get_tsc_train_dataset(
        dataset_name=dataset_name,
        data_ratio=atk_data_ratio,
        data_type='atk'
    )
    # Preprocess data
    x_train_atk, y_train_atk, x_test_atk, y_test_atk, enc = preprocess_data(
        x_train=x_train_atk,
        y_train=y_train_atk,
        x_test=x_test_atk,
        y_test=y_test_atk,
    )
    input_shape = x_train_atk.shape[1:]
    nb_classes = enc.categories_[0].shape[0]
    logger.info(f"[Attacker Dataset] Data preprocessed. Training samples: {len(x_train_atk)}, Test samples: {len(x_test_atk)}")

    if dataset_name != "iAWE":
        # Get sp dataset
        x_train_sp, y_train_sp, x_test_sp, y_test_sp = get_tsc_train_dataset(
            dataset_name=dataset_name,
            data_ratio=sp_data_ratio,
            data_type='sp'
        )

        # Preprocess sp data
        x_train_sp, y_train_sp, x_test_sp, y_test_sp, _ = preprocess_data(
            x_train=x_train_sp,
            y_train=y_train_sp,
            x_test=x_test_sp,
            y_test=y_test_sp,
        )
        logger.info(f"[Service Provider Dataset] dataset loaded. Training samples: {len(x_train_sp)}, Test samples: {len(x_test_sp)}")
    else:
        # For iAWE, we use the same data for both attacker and service provider
        x_train_sp, y_train_sp, x_test_sp, y_test_sp = x_train_atk, y_train_atk, x_test_atk, y_test_atk

    # Load the target classifier
    target_model_wrapper = ClassifierWrapper(
        output_directory=None, # This will be set later
        input_shape=input_shape,
        nb_classes=nb_classes,
        training_config=target_clf_config,
        clf_name=target_clf_name,
        verbose=True
    )
    target_model_wrapper.model.load_weights(target_clf_dir)
    target_model_wrapper.model.trainable = False

    # Load the surro classifier
    surro_model_wrapper = ClassifierWrapper(
        output_directory=None,
        input_shape=input_shape,
        nb_classes=nb_classes,
        training_config=surro_clf_config,
        clf_name=surro_clf_name,
        verbose=True
    )
    surro_model_wrapper.model.load_weights(surro_clf_dir)
    surro_model_wrapper.model.trainable = False

    # Create trigger generator
    noise_generator = TriggerGenerator(
        output_directory=None,
        generator_config=generator_config,
        max_amplitude=amplitude,
        input_shape=input_shape,
        enc=enc
    )
    noise_generator.generator.load_weights(gen_dir)
    noise_generator.generator.trainable = False

    # Test the generator and the target classifier
    x_triggered = noise_generator.apply_trigger(x_train_atk)
    y_targets = enc.transform(np.array([y_target]*len(x_triggered)).reshape(-1, 1)).toarray()
    
    # Create triggered dataset
    triggered_dataset = create_dataset(x_triggered, y_targets, 
                                       batch_size=target_clf_config["batch_size"],
                                       prefetch=True,
                                       shuffle=False).cache()
    
    test_atk_dataset = create_dataset(x_test_sp, y_test_sp,
                                    batch_size=target_clf_config["batch_size"],
                                    prefetch=True,
                                    shuffle=False).cache()

    test_sp_dataset = create_dataset(x_test_sp, y_test_sp,
                                    batch_size=surro_clf_config["batch_size"],
                                    prefetch=True,
                                    shuffle=False).cache()

    # Evaluate the target classifier on the poisoned samples
    target_atk_results = target_model_wrapper.model.evaluate(triggered_dataset, verbose=0)
    loss_target, asr_target = target_atk_results[0], target_atk_results[1]
    target_ca = target_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)[1]
    logger.info(f"[Initial ASR] Target '{target_clf_name}' - CA: {target_ca:.4f}, ASR: {asr_target:.4f}, ASR Loss: {loss_target:.4f}")

    # Evaluate the surrogate classifier on the poisoned samples
    surro_results = surro_model_wrapper.model.evaluate(triggered_dataset, verbose=0)
    loss_surro, asr_surro = surro_results[0], surro_results[1]
    surro_ca = surro_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)[1]
    logger.info(f"[Initial ASR] Surrogate '{surro_clf_name}' - CA: {surro_ca:.4f}, ASR: {asr_surro:.4f}, ASR Loss: {loss_surro:.4f}")

    # Log initial metrics to CSV
    csv_data = [
        0,
        f"{asr_surro:.4f}",
        f"{asr_target:.4f}",
        f"{surro_ca:.4f}",
        f"{target_ca:.4f}",
        f"{loss_surro:.4f}",
        f"{loss_target:.4f}",
        0.0000,  # FT Train Loss
        0.0000,  # FT Val Loss
        0
    ]
    log_to_csv(csv_file_path, csv_data)

    # Data storage for continuous updates with epoch-specific prefixes
    atk_storage = DataStorage(logger, "atk")
    sp_storage = DataStorage(logger, "sp")
    budget_counter = SimpleBudgetCounter()

    # Create datasets for training and testing
    x_train_gen, _ = select_stratified_by_ratio(
        x_train_atk, 
        y_train_atk, 
        data_ratio=gen_data_ratio
    )
    x_test_gen, _ = select_stratified_by_ratio(
        x_test_atk, 
        y_test_atk, 
        data_ratio=gen_data_ratio
    )

    # Ensure the generator data is reshaped correctly
    y_train_targets = enc.transform(np.array([y_target]*len(x_train_gen)).reshape(-1, 1)).toarray()
    y_test_targets = enc.transform(np.array([y_target]*len(x_test_gen)).reshape(-1, 1)).toarray()
    train_gen_dataset = create_dataset(x_train_gen, y_train_targets, 
                                    batch_size=generator_config["batch_size"],
                                    shuffle=True, prefetch=True).cache()
    val_gen_dataset = create_dataset(x_test_gen, y_test_targets, 
                                    batch_size=generator_config["batch_size"],
                                    shuffle=False, prefetch=True).cache()
    
    # Create finetuner
    finetuner = SurrogateModelFinetuner(
        input_shape=input_shape,
        nb_classes=nb_classes,
        verbose=True
    )

    # Setting up storage prefixes
    atk_storage.set_name_prefix(f"atk_storage")
    sp_storage.set_name_prefix(f"sp_storage")

    for main_epoch in range(1, main_epochs+1):
        # Main epoch logging
        logger.info('=' * 20 + f' Starting Main Epoch Training {main_epoch}/{main_epochs} ' + '=' * 20)
        epoch_out_dir = os.path.join(main_out_dir, f"epoch_{main_epoch+1}")
        if not os.path.exists(epoch_out_dir):
            os.makedirs(epoch_out_dir)

        # Reset the storage for the current epoch
        budget_counter.reset()
        atk_storage.reset()
        sp_storage.reset()

        # Attacker's model update
        ft_train_loss = 0.0
        ft_val_loss = 0.0
        if atk_update_interval > 0:
            if main_epoch > 0 and main_epoch % atk_update_interval == 0:
                # Getting the probabilities from the target model
                logger.info(f"[Surrogate Finetune] Getting the probabilities from the model, with {finetune_budget} samples...")
                ft_index = np.random.choice(
                    np.arange(len(x_train_atk)), 
                    size=finetune_budget, 
                    replace=False
                )
                x_ft = x_train_atk[ft_index]
                y_probs_ft = target_model_wrapper.model.predict(x_ft, batch_size=target_clf_config.get("batch_size", 128), verbose=0)

                # Query the budget counter for the fine-tuning budget
                budget_counter.query(finetune_budget)
                atk_storage.add_collected_data(
                    x=x_ft, 
                    y=y_probs_ft,
                )
                sp_storage.add_collected_data(
                    x=x_ft, 
                    y=y_probs_ft
                )

                for atk_epoch in range(1, atk_epochs+1):
                    logger.info(f"[Attack Epoch] Epoch {atk_epoch}/{atk_epochs} - Black-box generator training...")

                    # Preparing the data for finetuning
                    if atk_storage.has_previous_data():
                        # Get previous attack data
                        x_collected, y_collected = atk_storage.get_collected_data_probs()
                        x_surro_ft_train, x_surro_ft_test, y_surro_ft_train, y_surro_ft_test = train_test_split(
                            x_collected, y_collected, 
                            test_size=0.2, 
                            random_state=SEED
                        )
                        budget_counter.query(len(x_surro_ft_train))
                    else:
                        raise ValueError("No previous attack data available for finetuning surrogate model.")

                    # Finetune the surrogate model
                    logger.info(f"[Surrogate Finetune] Finetuning surrogate model {surro_clf_name} with train: {len(x_surro_ft_train)} / test: {len(x_surro_ft_test)} test...")
                    surro_ft_dir = os.path.join(epoch_out_dir, f'surro_ft_epoch_{atk_epoch}')
                    if not os.path.exists(surro_ft_dir):
                        os.makedirs(surro_ft_dir)
                    surro_model_wrapper, history = finetuner.finetune_surrogate(
                        x_train=x_surro_ft_train,
                        y_train_probs=y_surro_ft_train,
                        x_test=x_surro_ft_test,
                        y_test_probs=y_surro_ft_test,
                        log_dir=surro_ft_dir,
                        surro_model_wrapper=surro_model_wrapper,
                        epochs=finetune_params["epochs"],
                        batch_size=finetune_params["batch_size"],
                        lr=finetune_params["learning_rate"],
                        loss_type=ft_loss
                    )

                    # Extract finetune losses
                    ft_train_loss = history.history['loss'][-1]
                    if 'val_loss' in history.history and len(history.history['val_loss']) > 0:
                        ft_val_loss = history.history['val_loss'][-1]
                    else:
                        ft_val_loss = None
                    logger.info(f"[Surrogate Finetune] Surrogate model training loss: {ft_train_loss} - Validation loss: {ft_val_loss}\n")

                    # Set the surrogate model wrapper for the noise generator
                    logger.info(f"[Generator] Training the trigger generator with train: {len(x_train_atk)} / val: {len(x_test_atk)}...")
                    noise_generator.output_directory = os.path.join(epoch_out_dir, f"generator_epoch_{atk_epoch}")
                    if not os.path.exists(noise_generator.output_directory):
                        os.makedirs(noise_generator.output_directory)

                    # Train the noise generator
                    noise_generator.train_generator_full_model(
                        train_dataset=train_gen_dataset,
                        val_dataset=val_gen_dataset,
                        surro_model_wrapper=surro_model_wrapper,
                        epochs=generator_config["generator_epochs"] // atk_epochs,
                        logger=logger,
                        amplitude_reg_weight=amplitude_reg_weight
                    )

                    # If this is not the last attack epoch, inject poisoned data
                    if atk_epoch < atk_epochs:
                        logger.info(f"[Attack Epoch] Injecting poisoned data into the target model training set {inject_budget}...")
                        # Select a subset of the training data to poison
                        poison_indices = np.random.choice(len(x_train_atk), inject_budget, replace=False)
                        x_to_poison = x_train_atk[poison_indices]
                        x_triggered = noise_generator.apply_trigger(x_to_poison)

                        # Inject the poisoned data into the training set of the target model
                        y_pred_triggered_target = target_model_wrapper.model.predict(x_triggered, verbose=0)

                        # After evaluation, store the triggered data
                        budget_counter.query(len(y_pred_triggered_target))  
                        atk_storage.add_collected_data(
                            x=x_triggered, 
                            y=y_pred_triggered_target
                        )
                        sp_storage.add_collected_data(
                            x=x_triggered, 
                            y=y_pred_triggered_target
                        )
                        budget_counter.query(len(x_triggered))

                        # Finish the attack epoch
                        logger.info(f"[Attack Epoch] Attack training epoch {atk_epoch}/{atk_epochs} completed.\n")
            else:
                # Skip the attack because atk_update_interval is set to 0
                logger.info(f"[Attack Epoch] Skipping attack epoch training as atk_update_interval is set to 0.")

        # --- Inject the poisoned data into the training set of the target model ---
        if do_injection:
            logger.info(f"[Main Epoch] Injecting poisoned data into the target model training set {inject_budget}...")
            # Select a subset of the training data to poison
            poison_indices = np.random.choice(len(x_train_atk), inject_budget, replace=False)
            x_to_poison = x_train_atk[poison_indices]
            x_triggered = noise_generator.apply_trigger(x_to_poison)
            y_pred_triggered_target = target_model_wrapper.model.predict(x_triggered, verbose=0)
            # After evaluation, store the triggered data
            budget_counter.query(len(y_pred_triggered_target))
            atk_storage.add_collected_data(
                x=x_triggered,
                y=y_pred_triggered_target
            )
            sp_storage.add_collected_data(
                x=x_triggered,
                y=y_pred_triggered_target
            )
        else:
            logger.info("[Main Epoch] Injection is disabled. No poisoned data will be injected into the target model.")

        # --- Evaluate the attack after target model and surrogate model on clean and triggered data ---
        poison_indices = np.random.choice(len(x_train_atk), inject_budget, replace=False)
        x_triggered = noise_generator.apply_trigger(x_train_atk[poison_indices])
        y_targets = enc.transform(np.array([y_target]*len(x_triggered)).reshape(-1, 1)).toarray()
        test_triggered_dataset = create_dataset(x_triggered, y_targets, batch_size=512, shuffle=False, prefetch=True).cache()

        # Evaluate both models on clean and triggered data
        target_ca = target_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)[1]
        loss_target, asr_target = target_model_wrapper.model.evaluate(test_triggered_dataset, verbose=0)
        surro_ca = surro_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)[1]
        loss_surrogate, asr_surrogate = surro_model_wrapper.model.evaluate(test_triggered_dataset, verbose=0)

        # Get prediction distributions
        y_pred_triggered_target = target_model_wrapper.model.predict(test_triggered_dataset, verbose=0)
        y_pred_triggered_surro = surro_model_wrapper.model.predict(test_triggered_dataset, verbose=0)
        target_dist = np.unique(np.argmax(y_pred_triggered_target, axis=1), return_counts=True)
        surro_dist = np.unique(np.argmax(y_pred_triggered_surro, axis=1), return_counts=True)

        # Log results
        logger.info(f"[ASR Surro '{surro_clf_name}'] CA: {surro_ca:.4f}, ASR: {asr_surrogate:.4f}, Loss: {loss_surrogate:.4f}, Dist: {surro_dist}")
        logger.info(f"[ASR Target '{target_clf_name}'] CA: {target_ca:.4f}, ASR: {asr_target:.4f}, Loss: {loss_target:.4f}, Dist: {target_dist}")

        # Log the metrics to CSV
        csv_data = [
            main_epoch,
            f"{asr_surrogate:.4f}",
            f"{asr_target:.4f}",
            f"{surro_ca:.4f}",
            f"{target_ca:.4f}",
            f"{loss_surrogate:.4f}",
            f"{loss_target:.4f}",
            f"{ft_train_loss:.4f}",
            f"{ft_val_loss:.4f}",
            budget_counter.get_count()
        ]
        log_to_csv(csv_file_path, csv_data)

        # ----- Update the target model and surrogate model with the collected data -----
        if sp_storage.has_previous_data():
            x_collected, y_collected = sp_storage.get_collected_data()
            x_collected_train, x_collected_test, y_collected_train, y_collected_test = train_test_split(
                x_collected, y_collected, test_size=0.1, random_state=main_epoch + SEED
            )

            # Concatenate the collected data with the training set
            x_update_target_train = np.concatenate((x_train_sp.copy(), x_collected_train), axis=0)
            y_update_target_train = np.concatenate((y_train_sp.copy(), y_collected_train), axis=0)
            x_update_target_test = np.concatenate((x_test_sp.copy(), x_collected_test), axis=0)
            y_update_target_test = np.concatenate((y_test_sp.copy(), y_collected_test), axis=0)
        else:
            x_update_target_train = x_train_sp.copy()
            y_update_target_train = y_train_sp.copy()
            x_update_target_test = x_test_sp.copy()
            y_update_target_test = y_test_sp.copy()

        # Set the output directory for the target model update
        target_model_wrapper.output_directory = os.path.join(epoch_out_dir, f'target_model_update')
        if not os.path.exists(target_model_wrapper.output_directory):
            os.makedirs(target_model_wrapper.output_directory)

        # Train the target model with the collected data
        logger.info(f"[Main Epoch] Updating target model with {x_update_target_train.shape} train samples and {x_update_target_test.shape} test samples...")
        target_model_wrapper.model.trainable = True
        target_model_wrapper.train(
            x_train=x_update_target_train,
            y_train=y_update_target_train,
            x_val=x_update_target_test,
            y_val=y_update_target_test,
            batch_size=surro_clf_config['batch_size'],
            epochs=surro_clf_config['epochs']
        )

        # Concatenate the collected data with the surro model training set
        if atk_storage.has_previous_data():
            x_collected, y_collected = atk_storage.get_collected_data()
            x_collected_train, x_collected_test, y_collected_train, y_collected_test = train_test_split(
                x_collected, y_collected, test_size=0.1, random_state=main_epoch + SEED
            )

            # Concatenate the collected data with the training set
            x_update_surro_train = np.concatenate((x_train_atk.copy(), x_collected_train), axis=0)
            y_update_surro_train = np.concatenate((y_train_atk.copy(), y_collected_train), axis=0)
            x_update_surro_test = np.concatenate((x_test_atk.copy(), x_collected_test), axis=0)
            y_update_surro_test = np.concatenate((y_test_atk.copy(), y_collected_test), axis=0)
        else:
            x_update_surro_train = x_train_atk.copy()
            y_update_surro_train = y_train_atk.copy()
            x_update_surro_test = x_test_atk.copy()
            y_update_surro_test = y_test_atk.copy()

        # Also update the surro model if required
        logger.info(f"[Main Epoch] Updating surro model with {len(x_update_surro_train)} train samples and {len(x_update_surro_test)} test samples...")
        surro_model_wrapper.output_directory = os.path.join(epoch_out_dir, f'surro_model_update')
        if not os.path.exists(surro_model_wrapper.output_directory):
            os.makedirs(surro_model_wrapper.output_directory)
        surro_model_wrapper.model.trainable = True
        surro_model_wrapper.train(
            x_train=x_update_surro_train,
            y_train=y_update_surro_train,
            x_val=x_update_surro_test,
            y_val=y_update_surro_test,
            batch_size=surro_clf_config['batch_size'],
            epochs=surro_clf_config['epochs']
        )

        # Evaluate the updated models
        target_eval_updated = target_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)
        logger.info(f"[Updated Target Model] Clean accuracy on test: {target_eval_updated[1]:.4f}")
        surrogate_eval_updated = surro_model_wrapper.model.evaluate(test_sp_dataset, verbose=0)
        logger.info(f"[Updated Surrogate Model] Clean accuracy on test: {surrogate_eval_updated[1]:.4f}")
        # Log the end of the main epoch
        logger.info(f"[Main Epoch] Main epoch {main_epoch}/{main_epochs} completed.\n")
        
    logger.info(f"\n[COMPLETE] Continuous backdoor attack finished!")
    logger.info(f"[COMPLETE] Results saved to: {csv_file_path}")
    logger.info(f"[COMPLETE] Logs saved to: {log_file}")