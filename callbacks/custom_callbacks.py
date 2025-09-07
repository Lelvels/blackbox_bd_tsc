import keras
import numpy as np
import os

class GeneratorCheckpoint(keras.callbacks.Callback):
    """
    Custom callback to save only the generator model during the training of a composite model.
    
    This callback monitors a metric and saves the generator model when the metric improves.
    """
    def __init__(self, filepath, generator_model, monitor='val_loss', verbose=0,
                 save_best_only=True, mode='auto', save_freq='epoch'):
        super(GeneratorCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.generator_model = generator_model # The generator model to save
        self.save_best_only = save_best_only
        self.save_freq = save_freq
        
        if mode not in ['auto', 'min', 'max']:
            print(f"GeneratorCheckpoint mode '{mode}' is unknown, fallback to 'auto'.")
            mode = 'auto'
            
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.inf
            else:
                self.monitor_op = np.less
                self.best = np.inf

    def _get_file_path(self, epoch, logs):
        """Generate the file path, handling format strings safely."""
        try:
            # Create a safe logs dict with default values
            safe_logs = logs.copy() if logs else {}
            safe_logs.setdefault('epoch', epoch)
            
            # Add common metrics with default values if missing
            for metric in ['loss', 'val_loss', 'accuracy', 'val_accuracy']:
                if metric not in safe_logs:
                    safe_logs[metric] = 0.0
            
            # Format the filepath
            if '{' in self.filepath and '}' in self.filepath:
                formatted_path = self.filepath.format(epoch=epoch + 1, **safe_logs)
            else:
                formatted_path = self.filepath
                
            return formatted_path
        except (KeyError, ValueError) as e:
            print(f"Warning: Could not format filepath '{self.filepath}': {e}")
            # Fallback: append epoch to filename
            base, ext = os.path.splitext(self.filepath)
            return f"{base}_epoch_{epoch + 1}{ext}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Get the formatted file path
        filepath = self._get_file_path(epoch, logs)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        current = logs.get(self.monitor)
        if current is None:
            if self.verbose > 0:
                print(f"\nWarning: Metric '{self.monitor}' is not available in logs.")
                print(f"Available metrics: {list(logs.keys())}")
            
            # If monitor metric is not available, save anyway if not save_best_only
            if not self.save_best_only:
                if self.verbose > 0:
                    print(f"Saving generator model to {filepath}")
                self.generator_model.save(filepath)
            return

        if self.save_best_only:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}')
                    print(f'Saving generator model to {filepath}')
                self.best = current
                self.generator_model.save(filepath)
            else:
                if self.verbose > 0:
                    print(f'\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best:.5f}')
        else:
            if self.verbose > 0:
                print(f'\nEpoch {epoch + 1}: saving generator model to {filepath}')
            self.generator_model.save(filepath)

    def on_train_end(self, logs=None):
        """Save the final model at the end of training."""
        if self.verbose > 0:
            print("\nTraining completed. Saving final generator model.")
        
        # Save final model with '_final' suffix
        base, ext = os.path.splitext(self.filepath)
        final_path = f"{base}_final{ext}"
        
        # Remove format placeholders for final save
        if '{' in final_path:
            final_path = base.split('{')[0] + '_final' + ext
            
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        self.generator_model.save(final_path)
        
        if self.verbose > 0:
            print(f"Final generator model saved to {final_path}")