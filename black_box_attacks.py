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
from utils.utils import SimpleBudgetCounter, get_last_completed_epoch, load_model_if_exists, limit_gpu_memory
from utils.data_handler import get_tsc_train_dataset, preprocess_data, create_dataset, select_stratified_by_ratio
from models.clf_wrapper import ClassifierWrapper
from models.finetuner import SurrogateModelFinetuner 
from models.trigger_gen import TriggerGenerator
from utils.data_storage import DataStorage
from sklearn.model_selection import train_test_split
import joblib
# For reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Setup logger
def setup_logger(log_file=None):
    logger = logging.getLogger('black_box_attacks')
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

def initialize_csv_logger(csv_path, headers):
    """Initialize CSV file with headers"""
    if os.path.exists(csv_path):
        # If file exists, keep using it
        return
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

def log_to_csv(csv_path, data):
    """Append data row to CSV file"""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

limit_gpu_memory()
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Black-box Backdoor Attack on Time Series Classification')
    
    # === Basic Configuration ===
    parser.add_argument("--gpu", type=str, default='0', help="GPU to use for training (default: 0)")
    parser.add_argument("--dataset_name", type=str, default='iAWE', help="Dataset name")
    parser.add_argument("--exp_name", type=str, default='', help="Experiment name")
    parser.add_argument("--resume", action='store_true', help="Resume from last checkpoint")
    
    # === Model Configuration ===
    parser.add_argument("--target_clf_name", type=str, default='resnet', help="Target classifier name")
    parser.add_argument("--surrogate_clf_name", type=str, default='fcn', help="Surrogate classifier name")
    
    # === Attack Configuration ===
    parser.add_argument("--target_class", type=int, default=1, help="Target class for backdoor attack")
    parser.add_argument("--backdoor_training", action='store_true', help='If set, will train the backdoor attack with the surrogate model')
    parser.add_argument("--amplitude", type=float, default=0.2, help="Maximum amplitude for the trigger generator")
    parser.add_argument("--amplitude_reg_weight", type=float, default=1e-3, help="Amplitude regularization weight for the generator (default: 0.001)")

    # === Attack Budget Configuration ===
    parser.add_argument("--inject_budget", type=int, required=True, help="Injection budget, number of samples send to the target provider")
    parser.add_argument("--ft_budget", type=int, default=1000, help="Budget for fine-tuning the surrogate model (default: 1000)")
    
    # === Training Configuration ===
    parser.add_argument("--main_epochs", type=int, default=10, help="Main epochs for backdoor")
    parser.add_argument("--atk_epochs", type=int, default=2, help="Number of epochs for the attack")
    parser.add_argument("--ft_loss", type=str, default="kl_loss", help="Fine-tune loss")
    
    # === Data Configuration ===
    parser.add_argument('--atk_data_ratio', type=float, default=1.0, help='Ratio of data to use for attacker training')
    parser.add_argument("--sp_data_ratio", type=float, default=1.0, help='Ratio of data to use for service provider training')
    parser.add_argument("--generator_train_data_ratio", type=float, default=0.3, help="Ratio of data to use for generator training (default: 0.3)")
    
    args, unknown = parser.parse_known_args()

    # Set up the variables from command line arguments
    gpu = args.gpu
    dataset_name = args.dataset_name
    exp_name = args.exp_name
    resume = args.resume

    # Model Configuration
    target_clf_name = args.target_clf_name
    surro_clf_name = args.surrogate_clf_name

    # Attack Configuration
    y_target = args.target_class
    backdoor_training = args.backdoor_training
    max_amplitude = args.amplitude if args.amplitude > 0 else 0.2
    amplitude_reg_weight = args.amplitude_reg_weight if args.amplitude_reg_weight > 0 else 1e-3

    # Training Configuration
    main_epochs = args.main_epochs
    atk_epochs = args.atk_epochs
    ft_loss = args.ft_loss

    # Data Configuration
    atk_data_ratio = args.atk_data_ratio
    sp_data_ratio = args.sp_data_ratio
    generator_train_data_ratio = args.generator_train_data_ratio

    # Maximum queries for the attack
    ft_budget = args.ft_budget if args.ft_budget > 0 else 1000
    inject_budget = args.inject_budget
    total_queries = ft_budget + inject_budget*atk_epochs
    budget_counter = SimpleBudgetCounter()

    # Load configurations
    training_configs_path = os.path.join(SRC_DIR, "configs", "model_update.yaml")
    finetune_configs_path = os.path.join(SRC_DIR, "configs", "finetune_surrogate.yaml")
    generator_configs_path = os.path.join(SRC_DIR, "configs", "training_generator.yaml")
    
    training_configs = yaml.safe_load(open(training_configs_path, 'r'))
    finetune_configs = yaml.safe_load(open(finetune_configs_path, 'r'))
    generator_configs = yaml.safe_load(open(generator_configs_path, 'r'))

    # Configurations
    target_clf_config = training_configs[target_clf_name][dataset_name]
    surro_clf_config = training_configs[surro_clf_name][dataset_name]
    finetune_params = finetune_configs[surro_clf_name][dataset_name]
    generator_config = generator_configs["blackbox"][surro_clf_name][dataset_name]

    # Initialize logger
    if backdoor_training:
        main_out_dir = os.path.join(RESULT_DIR, dataset_name, "with_bd")
    else:
        main_out_dir = os.path.join(RESULT_DIR, dataset_name, "blackbox_adv")
    
    # With experiment name
    if exp_name == '':
        main_out_dir = os.path.join(main_out_dir, f"bb_{surro_clf_name}_{target_clf_name}_t{y_target}_amplitude_{max_amplitude}")
    else:
        main_out_dir = os.path.join(main_out_dir, f"bb_{surro_clf_name}_{exp_name}")
    if not os.path.exists(main_out_dir):
        os.makedirs(main_out_dir)
    log_file = os.path.join(main_out_dir, f"bb_{surro_clf_name}_{target_clf_name}.log")
    logger = setup_logger(log_file)

    # CSV Logger setup
    csv_headers = [
        'Main Epoch', 
        'Attack Epoch', 
        'Surrogate ASR', 
        'Target ASR', 
        'Surrogate Clean Acc',
        'Target Clean Acc',
        'Surrogate ASR Loss', 
        'Target ASR Loss',
        'Surrogate Finetune Train Loss', 
        'Surrogate Finetune Val Loss',
        'Query Count'
    ]
    csv_file_path = os.path.join(main_out_dir, f"attack_metrics_{surro_clf_name}_to_{target_clf_name}.csv")
    
    # Check for resuming from last checkpoint
    start_main_epoch = 1
    start_atk_epoch = 1
    resume = args.resume and os.path.exists(csv_file_path)

    # If resuming, load the last epoch numbers
    if resume:
        last_main_epoch, last_atk_epoch, resume = get_last_completed_epoch(csv_file_path, main_epochs, atk_epochs)
        if resume:
            if last_atk_epoch < atk_epochs:
                start_main_epoch = last_main_epoch
                start_atk_epoch = last_atk_epoch + 1
                logger.info(f"[RESUME] Resuming from Main Epoch {start_main_epoch}, Attack Epoch {start_atk_epoch}")
            else:
                start_main_epoch = last_main_epoch + 1
                start_atk_epoch = 1
                logger.info(f"[RESUME] Resuming from Main Epoch {start_main_epoch}, Attack Epoch {start_atk_epoch}")
        else:
            logger.info(f"[RESUME] No previous run found or training already complete")
            if last_main_epoch >= main_epochs:
                logger.info("Training already completed!")
                exit(0)
    else:
        logger.info("[RESUME] Resume mode not enabled, starting fresh training")
    
    initialize_csv_logger(csv_file_path, csv_headers)
    logger.info(f"[CSV Logger] Metrics will be saved to: {csv_file_path}")

    # Load finetune parameters
    if not finetune_params:
        logger.error(f"Finetune config not found for {surro_clf_name} with {target_clf_name} on {dataset_name}")
        raise ValueError(f"Finetune config not found for {surro_clf_name} with {target_clf_name} on {dataset_name}")
    
    # Log configurations
    logger.info("\n" + "=" * 100)
    logger.info("BLACK-BOX ATTACK CONFIGURATION")
    logger.info("=" * 100)
    
    # Basic attack settings
    logger.info(f"[Attack Setup] Dataset: {dataset_name}")
    logger.info(f"[Attack Setup] Target Class: {y_target}")
    logger.info(f"[Attack Setup] Target Classifier: {target_clf_name}")
    logger.info(f"[Attack Setup] Surrogate Classifier: {surro_clf_name}")
    logger.info(f"[Attack Setup] Experiment Name: {exp_name if exp_name else 'default'}")
    logger.info(f"[Attack Setup] GPU: {gpu}")
    
    # Data configuration
    logger.info(f"[Data Config] Attacker Data Ratio: {atk_data_ratio}")
    logger.info(f"[Data Config] Service Provider Data Ratio: {sp_data_ratio}")
    logger.info(f"[Data Config] Generator Training Data Ratio: {generator_train_data_ratio}")
    
    # Training configuration
    logger.info(f"[Training Config] Main Epochs: {main_epochs}")
    logger.info(f"[Training Config] Attack Epochs: {atk_epochs}")
    logger.info(f"[Training Config] Fine-tune Budget: {ft_budget}")
    logger.info(f"[Training Config] Injection Budget: {inject_budget}")
    logger.info(f"[Training Config] Total Queries per epoch: {total_queries}")
    logger.info(f"[Training Config] Fine-tune Loss: {ft_loss}")
    logger.info(f"[Training Config] Backdoor Training Enabled: {backdoor_training}")
    
    # Generator configuration
    logger.info(f"[Generator Config] Maximum Amplitude: {max_amplitude}")
    logger.info(f"[Generator Config] Amplitude Regularization Weight: {amplitude_reg_weight}")
    logger.info(f"[Generator Config] Batch Size: {generator_config['batch_size']}")
    logger.info(f"[Generator Config] Learning Rate: {generator_config.get('learning_rate', 'Not specified')}")
    logger.info(f"[Generator Config] Epochs: {generator_config.get('generator_epochs', 'Not specified') // atk_epochs}")
    
    # Detailed model configurations
    logger.info(f"[Target Model Config] {target_clf_config}")
    logger.info(f"[Surrogate Model Config] {surro_clf_config}")
    logger.info(f"[Finetune Config] {finetune_params}")
    
    # Resume information
    if resume:
        logger.info(f"[Resume] Resuming from Main Epoch: {start_main_epoch}, Last Attack Epoch: {last_atk_epoch}, Start Attack Epoch: {start_atk_epoch}")
    else:
        logger.info(f"[Resume] Starting fresh training")
    
    logger.info("=" * 100)
    
    # Save the used data if have to resume!
    used_data_folder = os.path.join(main_out_dir, "used_data")
    if not os.path.exists(used_data_folder):
        os.makedirs(used_data_folder)

    # Load datasets
    if not resume or start_main_epoch == 1:
        x_train_atk, y_train_atk, x_test_atk, y_test_atk = get_tsc_train_dataset(
            dataset_name=dataset_name,
            data_ratio=atk_data_ratio,
            data_type="atk"
        )

        # Preprocess the attacker data
        x_train_atk, y_train_atk, x_test_atk, y_test_atk, enc = preprocess_data(x_train_atk, y_train_atk, x_test_atk, y_test_atk)
        joblib.dump(enc, os.path.join(used_data_folder, "encoder.joblib"))

        if dataset_name != "iAWE":
            # Load the Service Provider dataset
            x_train_sp, y_train_sp, x_test_sp, y_test_sp = get_tsc_train_dataset(
                dataset_name=dataset_name,
                data_ratio=sp_data_ratio,
                data_type="sp"
            )
            x_train_sp, y_train_sp, x_test_sp, y_test_sp, _ = preprocess_data(x_train_sp, y_train_sp, x_test_sp, y_test_sp)
        else:
            # For iAWE, we use the same data for both attacker and service provider
            x_train_sp, y_train_sp, x_test_sp, y_test_sp = x_train_atk, y_train_atk, x_test_atk, y_test_atk

        # Save the datasets        
        np.save(os.path.join(used_data_folder, "x_train_atk.npy"), x_train_atk)
        np.save(os.path.join(used_data_folder, "y_train_atk.npy"), y_train_atk)
        np.save(os.path.join(used_data_folder, "x_test_atk.npy"), x_test_atk)
        np.save(os.path.join(used_data_folder, "y_test_atk.npy"), y_test_atk)

        # Save the service provider data
        np.save(os.path.join(used_data_folder, "x_train_sp.npy"), x_train_sp)
        np.save(os.path.join(used_data_folder, "y_train_sp.npy"), y_train_sp)
        np.save(os.path.join(used_data_folder, "x_test_sp.npy"), x_test_sp)
        np.save(os.path.join(used_data_folder, "y_test_sp.npy"), y_test_sp)
    else:
        # Load the attacker data
        x_train_atk = np.load(os.path.join(used_data_folder, "x_train_atk.npy"))
        y_train_atk = np.load(os.path.join(used_data_folder, "y_train_atk.npy"))
        x_test_atk = np.load(os.path.join(used_data_folder, "x_test_atk.npy"))
        y_test_atk = np.load(os.path.join(used_data_folder, "y_test_atk.npy"))

        # Load the service provider data
        x_train_sp = np.load(os.path.join(used_data_folder, "x_train_sp.npy"))
        y_train_sp = np.load(os.path.join(used_data_folder, "y_train_sp.npy"))
        x_test_sp = np.load(os.path.join(used_data_folder, "x_test_sp.npy"))
        y_test_sp = np.load(os.path.join(used_data_folder, "y_test_sp.npy"))

        # Load the encoder
        enc = joblib.load(os.path.join(used_data_folder, "encoder.joblib"))


    # Log dataset shapes and distributions
    atk_distribution = np.unique(np.argmax(y_train_atk, axis=1), return_counts=True)
    sp_distribution = np.unique(np.argmax(y_train_sp, axis=1), return_counts=True)
    logger.info(f"[Data] Encoder classes: {enc.categories_}")
    logger.info(f"[Data] Attacker data distribution: {y_train_atk.shape[0]} - {atk_distribution}")
    logger.info(f"[Data] Service provider data distribution: {y_train_sp.shape[0]} - {sp_distribution}")

    # Create datasets for training and testing
    train_atk_dataset = create_dataset(x_train_atk, y_train_atk, batch_size=generator_config["batch_size"], shuffle=True, prefetch=True).cache()
    test_atk_dataset = create_dataset(x_test_atk, y_test_atk, batch_size=generator_config["batch_size"], shuffle=False, prefetch=True).cache()
    train_sp_dataset = create_dataset(x_train_sp, y_train_sp, batch_size=generator_config["batch_size"], shuffle=True, prefetch=True).cache()
    test_sp_dataset = create_dataset(x_test_sp, y_test_sp, batch_size=generator_config["batch_size"], shuffle=False, prefetch=True).cache()
    
    # Create datasets for the generator
    x_train_gen, _ = select_stratified_by_ratio(
        x_train_atk, 
        y_train_atk, 
        data_ratio=generator_train_data_ratio
    )
    x_test_gen, _ = select_stratified_by_ratio(
        x_test_atk, 
        y_test_atk, 
        data_ratio=generator_train_data_ratio
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
    
    # Log dataset shapes
    input_shape = x_train_atk.shape[1:]
    nb_classes = enc.categories_[0].shape[0]
    logger.info(f"[Data] Input shape for models: {input_shape}, Number of classes: {nb_classes}, Target class: {y_target}")
    logger.info(f"[Data] Encoder categories: {enc.categories_}")
    
    # --- Initialize Target Model ---
    logger.info(f"[Setup] Initializing Target Model: {target_clf_name}")
    target_clf_train_config = training_configs.get(target_clf_name, {}).get(dataset_name, {})
    if not target_clf_train_config:
        logger.error(f"Training config not found for target model {target_clf_name} on dataset {dataset_name}")
        raise ValueError(f"Training config not found for target model {target_clf_name} on dataset {dataset_name}")
    target_model_wrapper = ClassifierWrapper(
        output_directory=None,  # Will be set during training
        input_shape=input_shape,
        nb_classes=nb_classes,
        training_config=target_clf_train_config,
        clf_name=target_clf_name,
        verbose=True,
        build=True
    )

    # Load target model weights - check for resume weights first
    if resume and start_main_epoch > 1:
        # Try to load from previous epoch's updated model
        resume_target_weights = os.path.join(main_out_dir, f"epoch_{start_main_epoch-1}", "target_model_update", "best_model.keras")
        if not load_model_if_exists(target_model_wrapper, resume_target_weights, logger):
            # Fallback to default weights
            default_target_model_dir = os.path.join(RESULT_DIR, dataset_name, target_clf_name, 'sp')
            target_model_weights_path = os.path.join(default_target_model_dir, 'best_model.keras')
            target_model_wrapper.model.load_weights(target_model_weights_path)
    else:
        default_target_model_dir = os.path.join(RESULT_DIR, dataset_name, target_clf_name, 'sp')
        target_model_weights_path = os.path.join(default_target_model_dir, 'best_model.keras')
        target_model_wrapper.model.load_weights(target_model_weights_path)

    default_target_model_dir = os.path.join(RESULT_DIR, dataset_name, target_clf_name, 'sp')
    target_model_weights_path = os.path.join(default_target_model_dir, 'best_model.keras')
    target_model_wrapper.model.load_weights(target_model_weights_path)

    # --- Initialize Surrogate Model (pre-finetuning) ---
    logger.info(f"[Setup] Initializing Surrogate Model: {surro_clf_name}")
    surrogate_clf_train_config = training_configs.get(surro_clf_name, {}).get(dataset_name, {})
    if not surrogate_clf_train_config:
        logger.error(f"Training config not found for surrogate model {surro_clf_name} on dataset {dataset_name}")
        raise ValueError(f"Training config not found for surrogate model {surro_clf_name} on dataset {dataset_name}")
    surro_model_wrapper = ClassifierWrapper(
        output_directory=None,
        input_shape=input_shape,
        nb_classes=nb_classes,
        training_config=surrogate_clf_train_config,
        clf_name=surro_clf_name,
        verbose=True,
        build=True
    )

    # Load surrogate model weights - check for resume weights first
    if resume and start_main_epoch > 1:
        # Try to load from previous epoch's updated model
        resume_surro_weights = os.path.join(main_out_dir, f"epoch_{start_main_epoch-1}", "surrogate_model_update", "best_model.keras")
        if not load_model_if_exists(surro_model_wrapper, resume_surro_weights, logger):
            # Fallback to default weights
            default_surrogate_model_dir = os.path.join(RESULT_DIR, dataset_name, surro_clf_name, 'atk')
            surrogate_model_weights_path = os.path.join(default_surrogate_model_dir, 'best_model.keras')
            surro_model_wrapper.model.load_weights(surrogate_model_weights_path)
    else:
        default_surrogate_model_dir = os.path.join(RESULT_DIR, dataset_name, surro_clf_name, 'atk')
        surrogate_model_weights_path = os.path.join(default_surrogate_model_dir, 'best_model.keras')
        surro_model_wrapper.model.load_weights(surrogate_model_weights_path)

    # --- Evaluate initial models ---
    target_eval_initial = target_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)
    logger.info(f"[Clean Acc Eval] Target '{target_clf_name}' clean accuracy on test: {target_eval_initial[1]:.4f}")
    surrogate_eval_initial = surro_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)
    logger.info(f"[Clean Acc Eval] Surrogate '{surro_clf_name}' clean accuracy on test: {surrogate_eval_initial[1]:.4f}")

    # Set the surrogate model wrapper for the noise generator
    finetuner = SurrogateModelFinetuner(
        input_shape=input_shape,
        nb_classes=nb_classes,
        verbose=True
    )

    # --- Prepare the noise generator ---
    noise_generator = TriggerGenerator(
        output_directory = None,  # Will be set during training
        generator_config = generator_config,
        max_amplitude = max_amplitude,
        input_shape = input_shape,
        enc = enc,
        gen_type = 'dynamic_ampl_cnn'
    )

    # Load noise generator weights - check for resume weights first
    if resume and start_main_epoch > 1:
        # Try to load from previous epoch's noise generator
        resume_gen_weights = os.path.join(main_out_dir, f"epoch_{last_main_epoch}", 
                                        f"generator_epoch_{last_atk_epoch}", 
                                        "best_generator.keras")
        if not os.path.exists(resume_gen_weights):
            logger.warning(f"[Resume Warning] Previous generator weights not found at {resume_gen_weights}, will start with new generator.")
        else:
            noise_generator.generator.load_weights(resume_gen_weights)
            logger.info(f"[Resume] Loaded noise generator weights from: {resume_gen_weights}")

    # -- Data storage for attack data --
    atk_storage = DataStorage(logger=logger, name_prefix="atk")
    sp_storage = DataStorage(logger=logger, name_prefix="sp")
    
    # -- Main training loop for backdoor attack --
    for main_epoch in range(start_main_epoch, main_epochs+1):
        # --- Preparing data for model-finetuning ---
        logger.info('=' * 20 + f' Starting Main Epoch Training {main_epoch}/{main_epochs} ' + '=' * 20)
        epoch_out_dir = os.path.join(main_out_dir, f"epoch_{main_epoch}")
        target_model_wrapper.model.trainable = False
        budget_counter.reset()
        atk_storage.reset()
        sp_storage.reset()

        # Getting the probability from the target model
        logger.info(f"[Surrogate Finetune] Getting the probabilities from the model, with {ft_budget} samples...")
        ft_index = np.random.choice(
            np.arange(len(x_train_atk)), 
            size=ft_budget, 
            replace=False
        )
        x_ft = x_train_atk[ft_index]
        y_probs_ft = target_model_wrapper.model.predict(x_ft, batch_size=target_clf_config.get("batch_size", 128), verbose=0)

        # Query the budget counter for the fine-tuning budget
        budget_counter.query(ft_budget)
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

            # Finetune the surrogate model
            logger.info(f"[Model Finetune] Finetuning surrogate model '{surro_clf_name}' with train: {len(x_surro_ft_train)} / test: {len(x_surro_ft_test)} test...")
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
            
            # Generate triggered samples for attack
            poison_indices = np.random.choice(len(x_train_atk), inject_budget, replace=False)
            x_triggered = noise_generator.apply_trigger(x_train_atk[poison_indices])
            y_targets = enc.transform(np.array([y_target]*len(x_triggered)).reshape(-1, 1)).toarray()
            test_triggered_dataset = create_dataset(x_triggered, y_targets, batch_size=512, shuffle=False, prefetch=True).cache()

            logger.info(f"[Target Model Attack] Generated {len(x_triggered)} triggered samples for evaluation...")

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
            logger.info(f"[ASR Target '{target_clf_name}'] CA: {target_ca:.4f}, ASR: {asr_target:.4f}, Loss: {loss_target:.4f}, Dist: {target_dist}")
            logger.info(f"[ASR Surro '{surro_clf_name}'] CA: {surro_ca:.4f}, ASR: {asr_surrogate:.4f}, Loss: {loss_surrogate:.4f}, Dist: {surro_dist}\n")

            # Update budget and storage
            budget_counter.query(len(y_pred_triggered_target))
            for storage in [atk_storage, sp_storage]:
                storage.add_collected_data(x=x_triggered, y=y_pred_triggered_target)

            # Log metrics to CSV
            csv_data = [
                main_epoch,
                atk_epoch,
                f"{asr_surrogate:.4f}",
                f"{asr_target:.4f}",
                f"{surro_ca:.4f}",
                f"{target_ca:4f}",
                f"{loss_surrogate:.4f}",
                f"{loss_target:.4f}",
                f"{ft_train_loss:.4f}",
                f"{ft_val_loss:.4f}",
                f"{budget_counter.get_count()}"
            ]            
            log_to_csv(csv_file_path, csv_data)
            # Finish the attack epoch
            logger.info(f"[Attack Epoch] Attack training epoch {atk_epoch}/{atk_epochs} completed.\n")

        # --- Update the target model and surrogate model with newly collected data ---
        if backdoor_training:
            logger.info(f"[Main Epoch] Backdoor training is enabled, updating models with collected data...")

            # Concatenate the collected data with the training set
            x_collected, y_collected = sp_storage.get_collected_data()
            y_class = np.argmax(y_collected, axis=1)
            x_collected_train, x_collected_test, y_collected_train, y_collected_test = train_test_split(
                x_collected, y_collected, test_size=0.1, random_state=main_epoch + SEED
            )
            # Concatenate the collected data with the training set
            x_update_target_train = np.concatenate((x_train_sp.copy(), x_collected_train), axis=0)
            y_update_target_train = np.concatenate((y_train_sp.copy(), y_collected_train), axis=0)
            x_update_target_test = np.concatenate((x_test_sp.copy(), x_collected_test), axis=0)
            y_update_target_test = np.concatenate((y_test_sp.copy(), y_collected_test), axis=0)

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
                batch_size=target_clf_config['batch_size'],
                epochs=target_clf_config['epochs']
            )

            # Concatenate the collected data with the surrogate model training set
            x_collected, y_collected = atk_storage.get_collected_data()
            y_class = np.argmax(y_collected, axis=1)
            x_collected_train, x_collected_test, y_collected_train, y_collected_test = train_test_split(
                x_collected, y_collected, test_size=0.1, random_state=main_epoch + SEED
            )
            # Concatenate the collected data with the training set
            x_update_surro_train = np.concatenate((x_train_atk.copy(), x_collected_train), axis=0)
            y_update_surro_train = np.concatenate((y_train_atk.copy(), y_collected_train), axis=0)
            x_update_surro_test = np.concatenate((x_test_atk.copy(), x_collected_test), axis=0)
            y_update_surro_test = np.concatenate((y_test_atk.copy(), y_collected_test), axis=0)

            # Set the output directory for the surrogate model update
            surro_model_wrapper.output_directory = os.path.join(epoch_out_dir, f'surrogate_model_update')
            if not os.path.exists(surro_model_wrapper.output_directory):
                os.makedirs(surro_model_wrapper.output_directory)
            
            # Also update the surrogate model with the collected data to make sure it align the target model
            logger.info(f"[Main Epoch] Updating surrogate model with {len(x_update_surro_train)} train samples and {len(x_update_surro_test)} test samples...")
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
            surrogate_eval_updated = surro_model_wrapper.model.evaluate(test_atk_dataset, verbose=0)
            logger.info(f"[Updated Surrogate Model] Clean accuracy on test: {surrogate_eval_updated[1]:.4f}")
            # Log the end of the main epoch
            logger.info(f"[Main Epoch] Main epoch {main_epoch}/{main_epochs} completed.\n")
        else:
            logger.info(f"[Main Epoch] Backdoor training is disabled, skipping model updates.")
        
    logger.info("\nScript finished.")