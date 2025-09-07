# Noise-GAN model
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import logging
import csv
from sklearn.preprocessing import OneHotEncoder

from utils.constants import SEED
from models.clf_wrapper import ClassifierWrapper
from models.gen_models import get_generator_by_name
from callbacks.custom_callbacks import GeneratorCheckpoint
import tensorboard

np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TriggerGenerator:
    def __init__(self, 
                 output_directory,
                 generator_config,
                 max_amplitude,
                 input_shape: np.array,
                 enc: OneHotEncoder,
                 gen_type='dynamic_ampl_cnn'):
        
        # Setup logging
        self.output_directory = output_directory
        self.generator_config = generator_config
        self.gen_type = gen_type.lower()
        self.enc = enc
        self.input_shape = input_shape
        self.max_amplitude = max_amplitude

        # Initialize the generator model
        self.generator = self.build_generator()
        return
    
    class CustomReduceLROnPlateau:
        def __init__(self, optimizer, monitor='val_accuracy', factor=0.5, patience=10, 
                        mode='max', min_lr=1e-6, verbose=True):
            self.optimizer = optimizer
            self.monitor = monitor
            self.factor = factor
            self.patience = patience
            self.mode = mode
            self.min_lr = min_lr
            self.wait = 0
            self.verbose = verbose
            self.best = float('-inf') if mode == 'max' else float('inf')
            self.current_lr = float(optimizer.learning_rate.numpy())
            
        def step(self, current_metric, epoch, logger=None):
            """
            Check if learning rate should be reduced based on current metric.
            
            Args:
                current_metric: Current value of the monitored metric
                epoch: Current epoch number
                logger: Optional logger for output
                
            Returns:
                bool: True if learning rate was reduced, False otherwise
            """
            if self.mode == 'max':
                improved = current_metric > self.best
            else:
                improved = current_metric < self.best
                
            if improved:
                self.best = current_metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                
                if self.wait >= self.patience and self.current_lr > self.min_lr:
                    old_lr = self.current_lr
                    self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                    self.optimizer.learning_rate.assign(self.current_lr)
                    self.wait = 0
                    
                    if self.verbose and logger:
                        logger.info(f"[LR Scheduler] Epoch {epoch + 1}: reducing learning rate from {old_lr:.6f} to {self.current_lr:.6f}")
                    
                    return True
            
            return False
    
    def build_generator(self) -> keras.Model:
        self.generator = get_generator_by_name(self.input_shape, self.max_amplitude, self.gen_type)
        return self.generator
    
    def apply_trigger(self, x_backdoor):
        # --- MODIFIED: Use both pattern and amplitude outputs ---
        # Get the pattern and amplitude from the generator model
        patterns, amplitudes = self.generator.predict(x_backdoor)
        
        # Ensure all arrays are float32 for consistency
        x_backdoor = x_backdoor.astype(np.float32)
        patterns = patterns.astype(np.float32)
        amplitudes = amplitudes.astype(np.float32)
        
        # Apply the trigger using the generated pattern and amplitude
        # Formula: x' = x * (1 + amplitude * pattern)
        x_backdoor_triggered = x_backdoor * (1.0 + amplitudes * patterns)
        
        return x_backdoor_triggered

    def train_generator(self, 
                        train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        surro_model_wrapper: ClassifierWrapper,
                        logger: logging.Logger,
                        batch_size: int = None,
                        epochs: int = None,
                        learning_rate: float = None,
                        amplitude_reg_weight: float = 0.0,
                        verbose=True):
        """
            Function to train the generator with custom training logic. 
            Keras is so bad in memory management that we have to implement everything from scratch to avoid memory issues.

        Args:
            train_dataset: tf.data.Dataset for training data
            val_dataset: tf.data.Dataset for validation data
            surro_model_wrapper: ClassifierWrapper for the surrogate model
            logger: Logger for logging training progress
            amplitude_reg_weight: Weight for the amplitude regularization term
            verbose: Whether to log training progress
        Returns:
            None

        """
        # Set up the model
        if batch_size is None:
            batch_size = self.generator_config.get('batch_size', 64)
        
        if epochs is None:
            epochs = self.generator_config.get('generator_epochs', 100)

        if learning_rate is None:
            learning_rate = self.generator_config.get('learning_rate', 0.001)
        
        # Initialize the generator model with regularization
        logger.info(f"[Generator] Training epochs: {epochs}, Batch size: {batch_size}, Max amplitude: {self.max_amplitude}, Amplitude regularization weight: {amplitude_reg_weight}, Generator type: {self.gen_type}")

        # Set up optimizer
        initial_lr = self.generator_config.get('learning_rate', 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=initial_lr)
        
        # Set up custom LR scheduler
        lr_scheduler = self.CustomReduceLROnPlateau(
            optimizer=optimizer,
            monitor='val_accuracy',
            factor=0.5,
            patience=max(1, epochs // 10),
            mode='max',
            min_lr=1e-6,
            verbose=True
        )
        
        # Freeze surrogate model
        surro_model_wrapper.model.trainable = False
        
        # Set up TensorBoard writer
        log_dir = os.path.join(self.output_directory, 'tensorboard_logs', 'generator_training')
        os.makedirs(log_dir, exist_ok=True)
        writer = tf.summary.create_file_writer(log_dir)
        
        # Custom training step
        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                # Generate trigger
                patterns, amplitudes = self.generator(x_batch, training=True)
                
                # Ensure all tensors are float32 for consistency
                x_batch = tf.cast(x_batch, tf.float32)
                patterns = tf.cast(patterns, tf.float32)
                amplitudes = tf.cast(amplitudes, tf.float32)
                
                # Apply trigger
                x_triggered = x_batch * (1.0 + amplitudes * patterns)
                
                # Get predictions from surrogate model
                predictions = surro_model_wrapper.model(x_triggered, training=False)
                
                # Calculate loss
                loss = keras.losses.categorical_crossentropy(y_batch, predictions)
                loss = tf.reduce_mean(loss)
                
                # Add amplitude regularization
                amp_reg = tf.reduce_mean(tf.square(tf.cast(amplitudes, tf.float32))) * tf.cast(amplitude_reg_weight, tf.float32)
                total_loss = loss + amp_reg
    
            # Get gradients and update generator
            gradients = tape.gradient(total_loss, self.generator.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
            
            return total_loss, loss, amp_reg, predictions

        # Simple training loop without callbacks but with TensorBoard logging
        best_val_acc = 0.0
        patience_counter = 0
        max_patience = epochs // 3
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            epoch_accuracies = []
            epoch_amp_regs = []
            
            for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
                total_loss, loss, amp_reg, predictions = train_step(x_batch, y_batch)
                
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(y_batch, axis=1), tf.argmax(predictions, axis=1)), 
                    tf.float32
                ))
                
                epoch_losses.append(total_loss.numpy())
                epoch_accuracies.append(accuracy.numpy())
                epoch_amp_regs.append(amp_reg.numpy())
            
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accuracies)
            avg_amp_reg = np.mean(epoch_amp_regs)
            
            # Validation
            val_losses = []
            val_accuracies = []
            val_amp_regs = []
            
            for x_val, y_val in val_dataset:
                x_val = tf.cast(x_val, tf.float32)
                patterns, amplitudes = self.generator(x_val, training=False)
                patterns = tf.cast(patterns, tf.float32)
                amplitudes = tf.cast(amplitudes, tf.float32)
                
                x_val_triggered = x_val * (1.0 + amplitudes * patterns)
                val_predictions = surro_model_wrapper.model(x_val_triggered, training=False)
                
                val_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_val, val_predictions))
                val_acc = tf.reduce_mean(tf.cast(
                    tf.equal(tf.argmax(y_val, axis=1), tf.argmax(val_predictions, axis=1)), 
                    tf.float32
                ))
                val_amp_reg = tf.reduce_mean(tf.square(amplitudes)) * amplitude_reg_weight
                
                val_losses.append(val_loss.numpy())
                val_accuracies.append(val_acc.numpy())
                val_amp_regs.append(val_amp_reg.numpy())
        
            avg_val_loss = np.mean(val_losses)
            avg_val_acc = np.mean(val_accuracies)
            avg_val_amp_reg = np.mean(val_amp_regs)

            # Update learning rate
            lr_scheduler.step(avg_val_acc, epoch, logger)
            
            # Log metrics to TensorBoard
            with writer.as_default():
                tf.summary.scalar('train/loss', avg_loss, step=epoch)
                tf.summary.scalar('train/accuracy', avg_acc, step=epoch)
                tf.summary.scalar('train/amplitude_regularization', avg_amp_reg, step=epoch)
                tf.summary.scalar('validation/loss', avg_val_loss, step=epoch)
                tf.summary.scalar('validation/accuracy', avg_val_acc, step=epoch)
                tf.summary.scalar('validation/amplitude_regularization', avg_val_amp_reg, step=epoch)
                tf.summary.scalar('learning_rate', optimizer.learning_rate, step=epoch)
                
                # Log histograms of the latest batch patterns and amplitudes
                if epoch % 5 == 0:  # Log histograms every 5 epochs to avoid too much data
                    patterns, amplitudes = self.generator(x_batch, training=False)
                    tf.summary.histogram('generator/patterns', patterns, step=epoch)
                    tf.summary.histogram('generator/amplitudes', amplitudes, step=epoch)
                
                writer.flush()
        
            if verbose:
                logger.info(f"[Generator] Epoch {epoch + 1}/{epochs} - "
                            f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, Train Amp Reg: {avg_amp_reg:.6f} - "
                            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, Val Amp Reg: {avg_val_amp_reg:.6f} - "
                            f"LR: {optimizer.learning_rate.numpy():.6f}")
            
            # Early stopping and model saving - Use weights instead of full model
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                patience_counter = 0
                # Save weights instead of full model to avoid Lambda layer serialization issues
                self.generator.save(os.path.join(self.output_directory, 'best_generator.keras'))
                if verbose:
                    logger.info(f"[Generator] New best validation accuracy: {best_val_acc:.4f} - Weights saved")
                
                # Log best model metrics
                with writer.as_default():
                    tf.summary.scalar('best/validation_accuracy', best_val_acc, step=epoch)
                    tf.summary.scalar('best/validation_loss', avg_val_loss, step=epoch)
                    writer.flush()
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                if verbose:
                    logger.info(f"[Generator] Early stopping at epoch {epoch + 1}")
                break
    
        # Close the TensorBoard writer
        writer.close()
        
        # Load best weights instead of full model
        best_weights_path = os.path.join(self.output_directory, 'best_generator.keras')
        if os.path.exists(best_weights_path):
            self.generator = keras.models.load_model(best_weights_path, compile=False, safe_mode=False)
            if verbose:
                logger.info(f"[Generator] Loaded best weights from {best_weights_path}")
    
        # Save final model using weights
        self.generator.save(os.path.join(self.output_directory, 'last_generator.keras'))
    
        if verbose:
            logger.info(f"[Generator] TensorBoard logs saved to: {log_dir}")

    def setup_callbacks(self, log_name_suffix="", 
                        monitor='val_accuracy', 
                        mode='max',
                        has_tensorboard=True,
                        has_reduce_lr=True,
                        has_checkpoint=True,
                        has_early_stopping=True,
                        checkpoint_dir=None):
        """
        Set up callbacks for model training with specialized configurations for generator vs backdoor models.
        
        Args:
            log_name_suffix: Optional suffix to add to the log directory name
            monitor: Metric to monitor for reducing LR and saving checkpoints
            mode: 'min' or 'max' for the monitored metric
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            List of callbacks configured for the current run
        """
        callbacks = []
        epochs = self.generator_config["generator_epochs"]
        # Create directories
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.output_directory, 'checkpoints', log_name_suffix)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set permissions for the checkpoint directory
        try:
            import stat
            os.chmod(checkpoint_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777 permissions
        except Exception as e:
            print(f"Could not set permissions for {checkpoint_dir}: {e}")
        
        # Create TensorBoard log directory
        log_dir = os.path.join(self.output_directory, 'tensorboard_logs', log_name_suffix)
        os.makedirs(log_dir, exist_ok=True)
        if has_tensorboard:
            # Basic TensorBoard callback - used for all models
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='epoch',
                profile_batch=0,
            )
            callbacks.append(tensorboard_callback)

        # Early stopping and learning rate reduction
        if has_early_stopping:
            # Add EarlyStopping callback
            early_stopping_callback = keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=int(epochs//3),  # Early stopping after 30% of epochs
                verbose=1,
                mode=mode,
                restore_best_weights=True
            )
            callbacks.append(early_stopping_callback)
        
        if has_reduce_lr:
            # Add ReduceLROnPlateau - different monitoring metrics based on model type
            reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,  # For generator: 'val_accuracy', For backdoor: 'clean_val_accuracy' 
                factor=0.5,
                patience=int(epochs//10),  # Reduce LR every 10 epochs
                min_lr=1e-6,
                verbose=1,
                mode=mode
            )

            callbacks.append(reduce_lr_callback)

        # Callbacks for saving the best model
        if has_checkpoint:
            # Add custom GeneratorCheckpoint callback for generator models
            generator_checkpoint = GeneratorCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_generator.keras'),
                generator_model=self.generator,  # Pass the model you want to save
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                verbose=1
            )
            callbacks.append(generator_checkpoint)

        return callbacks
    
    def get_full_model(self, backdoor_clf: keras.models.Model, gen_trainable=True, bd_trainable=True):
        # Set trainable flags
        backdoor_clf.trainable = bd_trainable
        self.generator.trainable = gen_trainable

        # -- MODIFIED: Handle two generator outputs ---
        original_input = self.generator.input
        pattern, amplitude = self.generator.outputs

        # A Note on the Trigger Application:
        def apply_trigger_layer(tensors):
            original, pat, amp = tensors
            # This is broadcastable: (batch, steps, feats) * (1 + (batch, 1, 1) * (batch, steps, 1))
            return original * (1.0 + amp * pat)

        # Use a Lambda layer to apply the trigger within the model graph
        poisoned_input = keras.layers.Lambda(apply_trigger_layer, name='trigger_application')([original_input, pattern, amplitude])

        # Create a new model with the generator and backdoor classifier
        final_out = backdoor_clf(poisoned_input)
        full_model = keras.models.Model(inputs=original_input, outputs=final_out)
        
        return full_model
    
    def train_generator_full_model(self, 
                        train_dataset: tf.data.Dataset,
                        val_dataset: tf.data.Dataset,
                        surro_model_wrapper: ClassifierWrapper,
                        logger: logging.Logger,
                        epochs: int = None,
                        batch_size: int = None,
                        learning_rate: float = None,
                        amplitude_reg_weight: float = 0.0):
        
        # Set up the model
        if batch_size is None:
            batch_size = self.generator_config.get('batch_size', 64)
        
        if epochs is None:
            epochs = self.generator_config.get('generator_epochs', 100)

        if learning_rate is None:
            learning_rate = self.generator_config.get('learning_rate', 0.001)
        
        # Begin training
        logger.info(f"[Generator] Training epochs: {epochs}, Batch size: {batch_size}, Generator type: {self.gen_type}, Max amplitude: {self.max_amplitude}, Amplitude regularization weight: {amplitude_reg_weight}")
        # Initialize the generator model with regularization
        full_model = self.get_full_model(surro_model_wrapper.model, True, False)
        full_model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        
        # Get callbacks for generator training
        generator_callbacks = self.setup_callbacks(
            monitor='val_accuracy',
            mode='max',
            has_reduce_lr=True,
            has_early_stopping=True,
            has_tensorboard=True,
            has_checkpoint=True,
            checkpoint_dir=self.output_directory
        )

        full_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            verbose=True,
            callbacks=generator_callbacks
        )

        # Save the generator model after training
        keras.models.save_model(self.generator, os.path.join(self.output_directory, 'last_generator.keras'))
        keras.backend.clear_session()  # Clear session to free memory
        logger.info(f"[Generator] Training complete. Best model saved to {self.output_directory}/best_generator.keras")

        return None