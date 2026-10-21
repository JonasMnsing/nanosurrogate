from nanosurrogate.env import setup_gpu_env
setup_gpu_env()

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

from nanosurrogate.data import load_and_prepare_data
from nanosurrogate.models import build_mlp

# ─── Configuration ────────────────────────────────────────────────────────────
FILEPATH      = "/home/jonasmensing/bagheera/data/1_funding_period/phase_space_sample/Nx=9_Ny=9_Ne=8.csv"
SCALE_FACTOR  = 0.16
BATCH_SIZE    = 1024
EPOCHS        = 500
LR            = 3e-4

CLASSIFIER_LAYERS = [[128, 1e-4, 0.1]] * 4  # Classifier braucht meist weniger Tiefe
REGRESSOR_LAYERS  = [[256, 1e-4, 0.0]] * 8
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1. Load Data
    data = load_and_prepare_data(FILEPATH, scale_factor=SCALE_FACTOR)
    
    # 2. Split (Train / Val)
    idx_train, idx_val = train_test_split(np.arange(len(data["X"])), test_size=0.2, random_state=42)
    
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(data["X"][idx_train])
    X_val   = x_scaler.transform(data["X"][idx_val])
    
    # =========================================================================
    # STEP 1: Train Classifier (All Data)
    # =========================================================================
    print("\n--- 1. Training Coulomb Blockade Classifier ---")
    y_c_train, y_c_val = data["y_class"][idx_train], data["y_class"][idx_val]
    
    steps_c = len(X_train) // BATCH_SIZE
    lr_c = CosineDecay(LR, decay_steps=steps_c * EPOCHS, alpha=0.1)
    
    clf = build_mlp(X_train.shape[1], CLASSIFIER_LAYERS, lr_c, is_classifier=True)
    clf.fit(
        X_train, y_c_train,
        validation_data=(X_val, y_c_val),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)],
        verbose=2
    )
    clf.save("classifier_model.keras")

    # =========================================================================
    # STEP 2: Train Regressor (ONLY Conducting Data + KMC Error Weights!)
    # =========================================================================
    print("\n--- 2. Training Current Regressor (Conducting Regime Only) ---")
    cond_train = (y_c_train.flatten() == 1.0)
    cond_val   = (y_c_val.flatten() == 1.0)
    
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_r_train = y_scaler.fit_transform(data["y_trans"][idx_train][cond_train])
    y_r_val   = y_scaler.transform(data["y_trans"][idx_val][cond_val])
    
    w_train = data["weights"][idx_train][cond_train]
    w_val   = data["weights"][idx_val][cond_val]
    
    steps_r = len(y_r_train) // BATCH_SIZE
    lr_r = CosineDecay(LR, decay_steps=steps_r * EPOCHS, alpha=0.1)
    
    reg = build_mlp(X_train.shape[1], REGRESSOR_LAYERS, lr_r, is_classifier=False)
    reg.fit(
        X_train[cond_train], y_r_train,
        sample_weight=w_train,                        # <--- KMC Fehler direkt als Gewichtung!
        validation_data=(X_val[cond_val], y_r_val, w_val),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)],
        verbose=2
    )
    reg.save("regressor_model.keras")
    
    with open("scalers.pkl", "wb") as f:
        pickle.dump({"x_scaler": x_scaler, "y_scaler": y_scaler, "scale_factor": SCALE_FACTOR}, f)
    print("Both models and scalers saved successfully!")

if __name__ == "__main__":
    main()