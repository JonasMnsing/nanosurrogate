"""
Training script for the Two-Stage Hurdle Surrogate Model (nanosurrogate).

Workflow:
---------
1. Loads KMC simulation data and separates Coulomb blockade (0) from conduction (1).
2. Performs a stratified 70/15/15 split (Train / Validation / Test) to preserve 
   the exact blockade-to-conduction ratio across all subsets.
3. Trains Stage 1 (Blockade Classifier) on ALL training samples using balanced 
   class weights to handle potential regime imbalance.
4. Trains Stage 2 (Current Regressor) EXCLUSIVELY on conducting samples (y_class == 1),
   weighted by inverse KMC simulation errors (clipped for gradient stability).
5. Exports trained models, fitted scalers, training histories, and the hold-out test set.
"""

from nanosurrogate.env import setup_gpu_env
setup_gpu_env()  # CRITICAL: Must be called before importing TensorFlow/Keras!

import pickle
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

from nanosurrogate.data import load_and_prepare_data
from nanosurrogate.models import build_mlp

# ─── Configuration ────────────────────────────────────────────────────────────
FILEPATH           = "data/Nx=9_Ny=9_Ne=8.csv"
OUTPUT_DIR         = Path("models/hurdle_baseline")

# Physical & Transformation Parameters
SCALE_FACTOR       = 0.16   # Transition point [aA] between linear and log behavior in arcsinh
BLOCKADE_THRESHOLD = 0.16   # Current cutoff [aA] (~1 e/s) defining Coulomb blockade

# Training Hyperparameters
BATCH_SIZE         = 1024
EPOCHS             = 500
LR                 = 3e-4
USE_ERROR_WEIGHTS  = True   # Toggle KMC inverse-error weighting for the regressor
WEIGHT_CLIP_RANGE  = (0.1, 10.0)  # Prevents extreme single-sample weights from exploding gradients

# Network Architectures: List of [nodes, l2_regularization, dropout_rate]
CLASSIFIER_LAYERS  = [[128, 1e-4, 0.1]] * 4  # 4 layers are sufficient for blockade hyperplane boundaries
REGRESSOR_LAYERS   = [[256, 1e-4, 0.0]] * 8  # 8 deep layers to capture non-linear tunneling landscape
# ──────────────────────────────────────────────────────────────────────────────


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # 1. Load Data & Stratified Train / Val / Test Split (70% / 15% / 15%)
    # =========================================================================
    print(f"Loading dataset from {FILEPATH}...")
    data = load_and_prepare_data(
        FILEPATH, 
        scale_factor=SCALE_FACTOR, 
        threshold=BLOCKADE_THRESHOLD
    )
    
    all_indices = np.arange(len(data["X"]))
    y_class_flat = data["y_class"].flatten()
    
    # First split: 70% Train, 30% Temp (Val + Test), stratified by blockade state
    idx_train, idx_temp = train_test_split(
        all_indices, test_size=0.30, random_state=42, stratify=y_class_flat
    )
    # Second split: Split Temp equally into 15% Val and 15% Test
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42, stratify=y_class_flat[idx_temp]
    )
    
    n_blocked = np.sum(y_class_flat == 0.0)
    print(f"Total samples: {len(all_indices)} | Blocked: {n_blocked} ({n_blocked/len(all_indices)*100:.1f}%)")
    print(f"Split sizes -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Fit Input Scaler strictly on Training Data to prevent data leakage
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(data["X"][idx_train])
    X_val   = x_scaler.transform(data["X"][idx_val])
    X_test  = x_scaler.transform(data["X"][idx_test])

    # =========================================================================
    # 2. STEP 1: Train Coulomb Blockade Classifier (All Data)
    # =========================================================================
    print("\n--- 1. Training Coulomb Blockade Classifier ---")
    y_c_train = data["y_class"][idx_train]
    y_c_val   = data["y_class"][idx_val]
    
    # Compute balanced class weights in case blocked states are rare in the dataset
    classes = np.unique(y_c_train.flatten())
    c_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_c_train.flatten())
    class_weight_dict = dict(zip(classes.astype(int), c_weights))
    print(f"Classifier class weights (0=Blocked, 1=Conducting): {class_weight_dict}")
    
    # Use ceil to accurately include the final partial batch in each epoch
    steps_c = int(np.ceil(len(X_train) / BATCH_SIZE))
    lr_c = CosineDecay(initial_learning_rate=LR, decay_steps=steps_c * EPOCHS, alpha=0.1)
    
    clf = build_mlp(X_train.shape[1], CLASSIFIER_LAYERS, lr_c, is_classifier=True)
    history_clf = clf.fit(
        X_train, y_c_train,
        validation_data=(X_val, y_c_val),
        class_weight=class_weight_dict,
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)],
        verbose=2
    )
    clf.save(OUTPUT_DIR / "classifier_model.keras")

    # =========================================================================
    # 3. STEP 2: Train Current Regressor (ONLY Conducting Data)
    # =========================================================================
    print("\n--- 2. Training Current Regressor (Conducting Regime Only) ---")
    # Boolean masks to filter out all blocked (0.0) states
    cond_train = (y_c_train.flatten() == 1.0)
    cond_val   = (y_c_val.flatten() == 1.0)
    
    # Fit Target Scaler ONLY on conducting training samples so 0.0 plateaus don't skew scaling
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_r_train = y_scaler.fit_transform(data["y_trans"][idx_train][cond_train])
    y_r_val   = y_scaler.transform(data["y_trans"][idx_val][cond_val])
    
    # Prepare sample weights from KMC errors (clipped to avoid gradient spikes)
    if USE_ERROR_WEIGHTS:
        w_train = np.clip(data["weights"][idx_train][cond_train], *WEIGHT_CLIP_RANGE)
        w_val   = np.clip(data["weights"][idx_val][cond_val], *WEIGHT_CLIP_RANGE)
        # Re-normalize after clipping so average weight remains 1.0
        w_train /= np.mean(w_train)
        w_val   /= np.mean(w_val)
        print(f"Using clipped KMC error weights (Range: [{w_train.min():.2f}, {w_train.max():.2f}])")
    else:
        w_train = None
        w_val   = None
        print("Training regressor with uniform sample weights (USE_ERROR_WEIGHTS=False).")
    
    steps_r = int(np.ceil(len(y_r_train) / BATCH_SIZE))
    lr_r = CosineDecay(initial_learning_rate=LR, decay_steps=steps_r * EPOCHS, alpha=0.1)
    
    reg = build_mlp(X_train.shape[1], REGRESSOR_LAYERS, lr_r, is_classifier=False)
    history_reg = reg.fit(
        X_train[cond_train], y_r_train,
        sample_weight=w_train,
        validation_data=(X_val[cond_val], y_r_val, w_val) if w_val is not None else (X_val[cond_val], y_r_val),
        batch_size=BATCH_SIZE, 
        epochs=EPOCHS,
        callbacks=[EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True)],
        verbose=2
    )
    reg.save(OUTPUT_DIR / "regressor_model.keras")

    # =========================================================================
    # 4. Save Artifacts, Histories & Hold-Out Test Set for Evaluation
    # =========================================================================
    with open(OUTPUT_DIR / "scalers.pkl", "wb") as f:
        pickle.dump({
            "x_scaler": x_scaler, 
            "y_scaler": y_scaler, 
            "scale_factor": SCALE_FACTOR,
            "blockade_threshold": BLOCKADE_THRESHOLD
        }, f)
        
    with open(OUTPUT_DIR / "training_history.pkl", "wb") as f:
        pickle.dump({
            "classifier": history_clf.history,
            "regressor": history_reg.history
        }, f)
        
    # Save untouched test split in raw and transformed formats for Jupyter evaluation
    np.savez(
        OUTPUT_DIR / "test_data.npz",
        X_test_raw=data["X"][idx_test],
        X_test_scaled=X_test,
        y_test_class=data["y_class"][idx_test],
        y_test_raw=data["y_raw"][idx_test],
        y_test_trans=data["y_trans"][idx_test],
        weights_test=data["weights"][idx_test]
    )
    
    print(f"\nTraining complete! All models, scalers, and test data saved to '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()