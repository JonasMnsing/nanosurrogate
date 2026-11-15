import pickle
import tensorflow as tf
from pathlib import Path

class PeriodicCheckpoint(tf.keras.callbacks.Callback):
    """
    Custom Keras Callback to save the model and its training history 
    at regular epoch intervals. Ideal for long HPC cluster runs to prevent 
    data loss during unexpected interruptions.
    """
    def __init__(self, filepath_base: str, save_every: int = 100):
        super().__init__()
        self.filepath_base = Path(filepath_base)
        self.save_every = save_every
        self.custom_history = {}

    def on_epoch_end(self, epoch, logs=None):
        # 1. Update the running history dictionary
        logs = logs or {}
        for k, v in logs.items():
            self.custom_history.setdefault(k, []).append(v)
            
        # 2. Save model and history at the specified interval
        if (epoch + 1) % self.save_every == 0:
            # Ensure the directory exists
            self.filepath_base.parent.mkdir(parents=True, exist_ok=True)
            
            # Save Model
            model_path = f"{self.filepath_base}_epoch_{epoch+1}.keras"
            self.model.save(model_path)
            
            # Save History
            hist_path = f"{self.filepath_base}_history_epoch_{epoch+1}.pkl"
            with open(hist_path, 'wb') as f:
                pickle.dump(self.custom_history, f)
                
            print(f"\n[Checkpoint] Saved model and history backup at Epoch {epoch + 1}")