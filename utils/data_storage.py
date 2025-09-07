import numpy as np
import logging
import os
from typing import Optional, Tuple

class DataStorage:
    def __init__(self, logger: Optional[logging.Logger] = None, name_prefix: str = ""):
        """
        Initialize DataStorage for collecting and managing attack data.
        
        Args:
            logger: Optional logger for debugging information
            name_prefix: Prefix for file names and log messages (e.g., "epoch_1_", "surrogate_")
        """
        self.logger = logger
        self.name_prefix = name_prefix
        self.reset()
    
    def reset(self):
        """Reset all stored data"""
        self.x_prev_atk = None
        self.y_prev_atk_probs = None
        self.y_prev_atk_pred = None
        self.x_collected = None
        self.y_collected = None
        self.y_collected_probs = None
        
        if self.logger:
            self.logger.info(f"[DataStorage {self.name_prefix}] Reset all collected data")
    
    def _probs_to_onehot(self, y_input: np.ndarray) -> np.ndarray:
        """
        Convert probability predictions or class indices to one-hot encoded format.
        
        Args:
            y_input: Either probability predictions of shape (n_samples, n_classes) 
                    or class indices of shape (n_samples,)
            
        Returns:
            One-hot encoded predictions
        """
        # Check if input is already probabilities (2D) or class indices (1D)
        if len(y_input.shape) == 1:
            # Input is class indices, convert to one-hot
            n_classes = len(np.unique(y_input))
            y_onehot = np.eye(n_classes)[y_input]
        elif len(y_input.shape) == 2:
            # Input is probabilities, convert to one-hot
            y_onehot = np.zeros_like(y_input)
            y_onehot[np.arange(len(y_input)), np.argmax(y_input, axis=1)] = 1
        else:
            raise ValueError(f"Invalid input shape: {y_input.shape}. Expected 1D or 2D array.")
        
        return y_onehot

    def add_collected_data(self, x: np.ndarray, y: np.ndarray):
        """
        Add new attack data (triggered samples and their predictions).
        
        Args:
            x: Collected samples of shape (n_samples, timesteps, features)
            y: Predictions or class indices of shape (n_samples,) or (n_samples, n_classes)
        """
        # Convert to probabilities if needed, then to one-hot
        if len(y.shape) == 1:
            # If class indices, we need to know the number of classes
            # For now, create a simple probability array (this might need adjustment)
            n_classes = len(np.unique(y))
            y_probs = np.eye(n_classes)[y]  # Convert to one-hot as probabilities
        else:
            y_probs = y  # Already probabilities
    
        y_pred_onehot = self._probs_to_onehot(y_probs)
        
        # Store previous attack data
        if self.x_prev_atk is None:
            self.x_prev_atk = x.copy()
            self.y_prev_atk_probs = y_probs.copy()
            self.y_prev_atk_pred = y_pred_onehot.copy()
        else:
            self.x_prev_atk = np.concatenate((self.x_prev_atk, x.copy()), axis=0)
            self.y_prev_atk_probs = np.concatenate((self.y_prev_atk_probs, y_probs.copy()), axis=0)
            self.y_prev_atk_pred = np.concatenate((self.y_prev_atk_pred, y_pred_onehot.copy()), axis=0)
        
        # Update collected data
        if self.x_collected is None:
            self.x_collected = x.copy()
            self.y_collected = y_pred_onehot.copy()
            self.y_collected_probs = y_probs.copy()
        else:
            self.x_collected = np.concatenate((self.x_collected, x.copy()), axis=0)
            self.y_collected = np.concatenate((self.y_collected, y_pred_onehot.copy()), axis=0)
            self.y_collected_probs = np.concatenate((self.y_collected_probs, y_probs.copy()), axis=0)
        
        if self.logger:
            self.logger.info(f"[DataStorage {self.name_prefix}] Added {len(x)} samples. "
                           f"Total collected: {len(self.x_collected)}")
    
    def get_previous_attack_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get previous attack data for fine-tuning.
        
        Returns:
            Tuple of (x_prev_atk, y_prev_atk_probs, y_prev_atk_pred)
        """
        return self.x_prev_atk, self.y_prev_atk_probs, self.y_prev_atk_pred
    
    def get_collected_data_probs(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get all collected data probabilities for model updates.
        
        Returns:
            Tuple of (x_collected, y_collected_probs)
        """
        return self.x_collected, self.y_collected_probs
    
    def get_collected_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get all collected data for model updates.
        
        Returns:
            Tuple of (x_collected, y_collected)
        """
        return self.x_collected, self.y_collected
    
    def has_previous_data(self) -> bool:
        """Check if there is any previous attack data"""
        return self.x_prev_atk is not None
    
    def has_collected_data(self) -> bool:
        """Check if there is any collected data"""
        return self.x_collected is not None
    
    def get_data_stats(self) -> dict:
        """
        Get statistics about stored data.
        
        Returns:
            Dictionary with data statistics
        """
        stats = {
            'name_prefix': self.name_prefix,
            'prev_atk_samples': len(self.x_prev_atk) if self.x_prev_atk is not None else 0,
            'collected_samples': len(self.x_collected) if self.x_collected is not None else 0,
            'has_previous_data': self.has_previous_data(),
            'has_collected_data': self.has_collected_data()
        }
        
        if self.y_prev_atk_pred is not None:
            # Get class distribution of previous predictions
            class_counts = np.sum(self.y_prev_atk_pred, axis=0)
            stats['prev_class_distribution'] = class_counts.tolist()
        
        if self.y_collected is not None:
            # Get class distribution of collected data
            class_counts = np.sum(self.y_collected, axis=0)
            stats['collected_class_distribution'] = class_counts.tolist()
        
        return stats
    
    def log_stats(self):
        """Log current data statistics"""
        if self.logger:
            stats = self.get_data_stats()
            self.logger.info(f"[DataStorage {self.name_prefix} Stats] {stats}")
    
    def clear_previous_data(self):
        """Clear only previous attack data, keep collected data"""
        self.x_prev_atk = None
        self.y_prev_atk_probs = None
        self.y_prev_atk_pred = None
        
        if self.logger:
            self.logger.info(f"[DataStorage {self.name_prefix}] Cleared previous attack data")
    
    def set_name_prefix(self, new_prefix: str):
        """
        Update the name prefix for this DataStorage instance.
        
        Args:
            new_prefix: New prefix to use for file names and log messages
        """
        self.name_prefix = new_prefix
        if self.logger:
            self.logger.info(f"[DataStorage] Updated name prefix to: {self.name_prefix}")
    
    def get_filename(self, base_name: str) -> str:
        """
        Generate filename with prefix.
        
        Args:
            base_name: Base filename without extension
            
        Returns:
            Filename with prefix applied
        """
        if self.name_prefix:
            return f"{self.name_prefix}{base_name}"
        return base_name
    
    def save_data(self, save_dir: str):
        """
        Save all collected data to numpy files with name prefix.
        
        Args:
            save_dir: Directory to save the data
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if self.x_collected is not None:
            x_filename = self.get_filename("x_collected.npy")
            y_filename = self.get_filename("y_collected.npy")
            np.save(os.path.join(save_dir, x_filename), self.x_collected)
            np.save(os.path.join(save_dir, y_filename), self.y_collected)
        
        if self.x_prev_atk is not None:
            x_prev_filename = self.get_filename("x_prev_atk.npy")
            y_prev_probs_filename = self.get_filename("y_prev_atk_probs.npy")
            y_prev_pred_filename = self.get_filename("y_prev_atk_pred.npy")
            np.save(os.path.join(save_dir, x_prev_filename), self.x_prev_atk)
            np.save(os.path.join(save_dir, y_prev_probs_filename), self.y_prev_atk_probs)
            np.save(os.path.join(save_dir, y_prev_pred_filename), self.y_prev_atk_pred)
        
        if self.logger:
            self.logger.info(f"[DataStorage {self.name_prefix}] Saved data to {save_dir}")
    
    def load_data(self, save_dir: str):
        """
        Load collected data from numpy files with name prefix.
        
        Args:
            save_dir: Directory to load the data from
        """
        # Load collected data
        x_collected_filename = self.get_filename("x_collected.npy")
        y_collected_filename = self.get_filename("y_collected.npy")
        x_collected_path = os.path.join(save_dir, x_collected_filename)
        y_collected_path = os.path.join(save_dir, y_collected_filename)
        
        if os.path.exists(x_collected_path) and os.path.exists(y_collected_path):
            self.x_collected = np.load(x_collected_path)
            self.y_collected = np.load(y_collected_path)
        
        # Load previous attack data
        x_prev_filename = self.get_filename("x_prev_atk.npy")
        y_prev_probs_filename = self.get_filename("y_prev_atk_probs.npy")
        y_prev_pred_filename = self.get_filename("y_prev_atk_pred.npy")
        x_prev_path = os.path.join(save_dir, x_prev_filename)
        y_prev_probs_path = os.path.join(save_dir, y_prev_probs_filename)
        y_prev_pred_path = os.path.join(save_dir, y_prev_pred_filename)
        
        if (os.path.exists(x_prev_path) and 
            os.path.exists(y_prev_probs_path) and 
            os.path.exists(y_prev_pred_path)):
            self.x_prev_atk = np.load(x_prev_path)
            self.y_prev_atk_probs = np.load(y_prev_probs_path)
            self.y_prev_atk_pred = np.load(y_prev_pred_path)
        
        if self.logger:
            self.logger.info(f"[DataStorage {self.name_prefix}] Loaded data from {save_dir}")
            self.log_stats()
    
    def save_data_with_custom_prefix(self, save_dir: str, custom_prefix: str):
        """
        Save data with a custom prefix (without changing the instance prefix).
        
        Args:
            save_dir: Directory to save the data
            custom_prefix: Custom prefix for this save operation only
        """
        original_prefix = self.name_prefix
        self.name_prefix = custom_prefix
        self.save_data(save_dir)
        self.name_prefix = original_prefix
    
    def load_data_with_custom_prefix(self, save_dir: str, custom_prefix: str):
        """
        Load data with a custom prefix (without changing the instance prefix).
        
        Args:
            save_dir: Directory to load the data from
            custom_prefix: Custom prefix for this load operation only
        """
        original_prefix = self.name_prefix
        self.name_prefix = custom_prefix
        self.load_data(save_dir)
        self.name_prefix = original_prefix