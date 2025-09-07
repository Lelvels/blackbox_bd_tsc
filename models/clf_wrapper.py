# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import os
import keras
import tensorflow as tf
import numpy as np
from utils.constants import SEED
from models.clf_models import get_model_by_name
from utils.data_handler import create_dataset
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ClassifierWrapper:
    def __init__(self, 
                 output_directory, 
                 input_shape, 
                 nb_classes, 
                 training_config,
                 clf_name='fcn', 
                 verbose=False, 
                 build=True, 
                 custom_loss=None):
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.training_config = training_config
        self.custom_loss = custom_loss  
        self.clf_name = clf_name
        if build == True:
            self.model = self.build_model()
            self.verbose = verbose
        return

    def build_model(self) -> keras.Model:
        """
        Build the model based on the specified model name and input shape.
        Args:
            input_shape (tuple): Shape of the input data.
            nb_classes (int): Number of classes for classification.
        Returns:
            keras.Model: Compiled model.
        """
        # Get the base model
        model = get_model_by_name(self.clf_name, self.input_shape, self.nb_classes)

        # Attach the model's loss, optimizer, and metrics
        if self.custom_loss is None:
            model.compile(loss='categorical_crossentropy', 
                          optimizer=keras.optimizers.Adam(learning_rate=self.training_config['learning_rate']),
                          metrics=['accuracy'])
        else:
            model.compile(loss=self.custom_loss, 
                          optimizer=keras.optimizers.Adam(learning_rate=self.training_config['learning_rate']), 
                          metrics=['accuracy'])

        return model
    
    def _setup_callbacks(self, epochs, logging_tb=False, reduce_lr=True, model_checkpoint=True, early_stopping=True):
        # Reduce learning rate
        callbacks = []
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        if logging_tb:
            # TensorBoard logging
            log_dir = os.path.join(self.output_directory, 'tensorboard_logs')
            tensorboard = keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
        
        if reduce_lr:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=epochs//10, min_lr=0.0001, verbose=1)
            callbacks.append(reduce_lr)
        
        if model_checkpoint:
            # Save the best model
            file_path = os.path.join(self.output_directory, 'best_model.keras')
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_accuracy', save_best_only=True, verbose=1)
            callbacks.append(model_checkpoint)

        # Add early stopping
        if early_stopping:
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=epochs//2, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping)
        
        return callbacks

    def train(self, 
              x_train, y_train, 
              x_val, y_val, 
              batch_size=None, epochs=None, 
              logging_tb=True, reduce_lr=True, 
              model_checkpoint=True, early_stopping=True):
        # Parameters
        if batch_size is None:
            batch_size = self.training_config['batch_size']
        if epochs is None:
            epochs = self.training_config['epochs']
        # Training the model
        if os.path.exists(os.path.join(self.output_directory, 'initial_model.keras')):
            print("Initial model already exists. Skipping initial save.")
        else:
            # Save the initial model
            if not os.path.exists(self.output_directory):
                os.makedirs(self.output_directory)
            print("Saving initial model to:", os.path.join(self.output_directory, 'initial_model.keras'))
        print("Training model with batch size:", batch_size, "and epochs:", epochs)
        callbacks = self._setup_callbacks(epochs, 
                                         logging_tb=logging_tb, 
                                         reduce_lr=reduce_lr, 
                                         model_checkpoint=model_checkpoint, 
                                         early_stopping=early_stopping)
        
        # Create tf.data.Dataset
        train_dataset = create_dataset(x_train, y_train, batch_size=batch_size, shuffle=True, prefetch=True).cache()
        val_dataset = create_dataset(x_val, y_val, batch_size=batch_size, shuffle=False, prefetch=True).cache()
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def evaluate_with_dataset(self, dataset):
        """Evaluate model using tf.data.Dataset"""
        if not isinstance(dataset, tf.data.Dataset):
            raise TypeError("dataset must be a tf.data.Dataset")
        
        return self.model.evaluate(dataset, verbose=0)
    
    def predict_with_dataset(self, dataset):
        """Predict using tf.data.Dataset"""
        if not isinstance(dataset, tf.data.Dataset):
            raise TypeError("dataset must be a tf.data.Dataset")
        
        predictions = []
        for batch in dataset:
            batch_pred = self.model.predict_on_batch(batch)
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)

    def get_logits(self, x_data, batch_size=512):
        """
        Get logits (pre-softmax outputs) from the model.
        
        Args:
            x_data: Input data with shape (batch_size, timesteps, features)
            
        Returns:
            Logits tensor with shape (batch_size, nb_classes)
        """
        # Create a new model that outputs from the Dense layer (before softmax activation)
        logits_model = keras.models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('logits').output
        )
        
        return logits_model.predict(x_data, batch_size=batch_size, verbose=0)