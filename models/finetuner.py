import os
import numpy as np
from keras import callbacks as callbacks
import keras
import tensorflow as tf
from utils.constants import SEED
from models.clf_wrapper import ClassifierWrapper
from utils.data_handler import create_dataset
import logging

# For reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

class SurrogateModelFinetuner:
    def __init__(self,
                 input_shape,
                 nb_classes,
                 verbose=False):
        """
        Initializes the SurrogateModelFinetuner.

        Args:
            input_shape (tuple): Shape of the input data.
            nb_classes (int): Number of output classes.
            surrogate_model_wrapper (ClassifierWrapper): Wrapper for the surrogate model.
            target_model_wrapper (ClassifierWrapper): Wrapper for the target model.
            surrogate_clf_name (str): Name of the surrogate classifier model.
            target_clf_name (str): Name of the target classifier model.
            verbose (bool): Verbosity mode for training.
        """
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.verbose = verbose

    def _setup_callbacks(self, log_dir, surro_model_wrapper: ClassifierWrapper):
        """
        Setup callbacks for training the surrogate model.
        
        Args:
            log_dir (str): Directory to save logs and model checkpoints.
            surro_model_wrapper (ClassifierWrapper): Wrapper for the surrogate model.

        Returns:
            list: List of Keras callbacks.
        """
        callbacks_list = []
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=surro_model_wrapper.verbose
        )
        
        # TensorBoard logging
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=os.path.join(log_dir, 'tensorboard_logs'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0
        )
        
        # Model checkpointing
        checkpoint_path = os.path.join(log_dir, 'best_model.keras')
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=surro_model_wrapper.verbose
        )
        
        callbacks_list.append(model_checkpoint)
        callbacks_list.append(tensorboard_callback)
        callbacks_list.append(reduce_lr)
        
        return callbacks_list

    def finetune_surrogate(self, 
                           x_train,
                           y_train_probs,
                           x_test,
                           y_test_probs,
                           log_dir,
                           surro_model_wrapper: ClassifierWrapper,
                           epochs, 
                           batch_size,
                           lr=0.01,
                           loss_type="kl_loss"):
        """
        Finetunes the surrogate model using knowledge distillation.
        
        Args:
            x_train (np.ndarray): Training input data.
            y_train_probs (np.ndarray): Target probability distributions for training.
            x_test (np.ndarray): Test input data.
            y_test_probs (np.ndarray): Target probability distributions for testing.
            log_dir (str): Directory to save logs and model checkpoints.
            surro_model_wrapper (ClassifierWrapper): Wrapper for the surrogate model.
            logger (logging.Logger): Logger for training progress.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Training batch size. Defaults to 128.
            lr (float, optional): Learning rate. Defaults to 0.01.
            loss_type (str, optional): Loss function type ('kl_loss' or 'mse_loss'). Defaults to 'kl_loss'.

        Returns:
            tuple: (finetuned_model_wrapper, training_history)
        """
        # --- Setup Callbacks ---
        callbacks_list = self._setup_callbacks(log_dir, surro_model_wrapper)

        # --- Changing loss function ---
        if loss_type == "kl_loss":
            if lr is None:
                lr = surro_model_wrapper.training_config.get('learning_rate', 0.001)
            surro_model_wrapper.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=keras.losses.KLDivergence(),
                metrics=[keras.metrics.KLDivergence()]
            )
        elif loss_type == "mse_loss":
            if lr is None:
                lr = surro_model_wrapper.training_config.get('learning_rate', 0.001)
            surro_model_wrapper.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.KLDivergence()]
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Use 'kl_loss' or 'mse_loss'.")
        
        # Create tf.data.Dataset objects for training and validation
        train_dataset = create_dataset(
            x_train, 
            y_train_probs, 
            batch_size=batch_size, 
            shuffle=True, 
            prefetch=True
        )

        test_dataset = create_dataset(
            x_test, 
            y_test_probs, 
            batch_size=batch_size, 
            shuffle=False, 
            prefetch=True
        )

        # --- Train the surrogate model ---
        surro_model_wrapper.model.trainable = True
        history = surro_model_wrapper.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks_list,
            shuffle=True
        )
            
        # --- Re-compile the surrogate model after training ---
        surro_model_wrapper.model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            metrics=['accuracy']
        )

        return surro_model_wrapper, history
