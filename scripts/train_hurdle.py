"""
Training script for the Two-Stage Hurdle Surrogate Model (nanosurrogate).

Workflow:
---------
1. Loads KMC simulation data, transforms currents via arcsinh(I / scale_factor),
   and separates Coulomb blockade (0) from conduction (1).
2. Expands raw electrode voltages with physics-informed features (pairwise potential
   differences, pairwise sums, and selected products).
3. Performs a stratified 70/15/15 split (Train / Validation / Test) to preserve
   the exact blockade-to-conduction ratio across all subsets.
4. Trains Stage 1 (Blockade Classifier) on all training samples using square-root
   damped balanced class weights (or bypasses Stage 1 if blockade ratio is negligible).
5. Trains Stage 2 (Current Regressor) exclusively on conducting samples (y_class == 1),
   weighted by clipped inverse KMC simulation errors and scheduled via ReduceLROnPlateau.
6. Exports trained models, fitted scalers, feature configurations, training histories,
   and the untouched hold-out test set.
"""

from nanosurrogate.env import setup_gpu_env
setup_gpu_env()  # CRITICAL: Must be called before importing TensorFlow/Keras!

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay

from nanosurrogate.data import add_physics_features, load_and_prepare_data
from nanosurrogate.models import build_mlp

# ─── Configuration ────────────────────────────────────────────────────────────
FILEPATH            = "data/phase_space_sample/Nx=3_Ny=3_Ne=8_self_cap0.00.csv"
OUTPUT_DIR          = Path("models/hurdle_baseline")
RANDOM_SEED         = 42

# Physical & Transformation Parameters
SCALE_FACTOR        = 1e6   # Transition scale [aA] between linear and log regime in arcsinh
BLOCKADE_THRESHOLD  = 0.16  # Current cutoff [aA] (~1 e/s) defining the Coulomb blockade regime

# Training Hyperparameters
BATCH_SIZE          = 256
EPOCHS              = 1000
LR                  = 5e-4
USE_ERROR_WEIGHTS   = True       # Toggle KMC inverse-error weighting for the regressor
WEIGHT_CLIP_RANGE   = (0.5, 2.0) # Prevents extreme single-sample weights from distorting loss

# Network Architectures: List of [nodes, l2_regularization, dropout_rate]
CLASSIFIER_LAYERS   = [[128, 1e-5, 0.05]] * 4  # 4 layers for smooth blockade hyperplane boundaries
REGRESSOR_LAYERS    = [[512, 1e-7, 0.0]] * 6   # 6 deep layers to capture non-linear tunneling currents

# Bypass Thresholds (skips Stage 1 Classifier if dataset is almost purely conducting)
MIN_BLOCKED_SAMPLES = 100
MIN_BLOCKED_RATIO   = 0.02
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """Executes the end-to-end training pipeline for the Hurdle Surrogate model."""
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1. Load Data, Engineer Physics Features & Stratified Split (70 / 15 / 15)
    # =========================================================================
    print(f"Loading dataset from {FILEPATH}...")
    data = load_and_prepare_data(
        FILEPATH,
        scale_factor=SCALE_FACTOR,
        threshold=BLOCKADE_THRESHOLD,
    )

    n_electrodes = data["X"].shape[1]
    all_pairs = list(combinations(range(n_electrodes), 2))
    prod_pairs = [(5, 6), (3, 4)]

    data["X"] = add_physics_features(
        data["X"],
        diff_pairs=all_pairs,
        sum_pairs=all_pairs,
        prod_pairs=prod_pairs,
    )
    print(f"Expanded input features from {n_electrodes} base electrodes to {data['X'].shape[1]} features.")

    all_indices = np.arange(len(data["X"]))
    y_class_flat = data["y_class"].flatten()

    # First split: 70% Train, 30% Temp (Val + Test), stratified by blockade state
    idx_train, idx_temp = train_test_split(
        all_indices, test_size=0.30, random_state=RANDOM_SEED, stratify=y_class_flat
    )
    # Second split: Split Temp equally into 15% Val and 15% Test
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_class_flat[idx_temp]
    )

    n_blocked = int(np.sum(y_class_flat == 0.0))
    print(f"Total samples: {len(all_indices)} | Blocked: {n_blocked} ({n_blocked / len(all_indices) * 100:.1f}%)")
    print(f"Split sizes -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Fit Input Scaler strictly on Training Data to prevent data leakage
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = x_scaler.fit_transform(data["X"][idx_train])
    X_val   = x_scaler.transform(data["X"][idx_val])
    X_test  = x_scaler.transform(data["X"][idx_test])

    # =========================================================================
    # 2. STAGE 1: Train Coulomb Blockade Classifier (or Bypass if negligible)
    # =========================================================================
    print("\n--- 1. Training Coulomb Blockade Classifier ---")
    y_c_train = data["y_class"][idx_train]
    y_c_val   = data["y_class"][idx_val]

    n_blocked_train = int(np.sum(y_c_train == 0.0))
    blocked_ratio   = n_blocked_train / len(y_c_train)

    if n_blocked_train < MIN_BLOCKED_SAMPLES or blocked_ratio < MIN_BLOCKED_RATIO:
        print(f"Only {n_blocked_train} blocked samples ({blocked_ratio * 100:.2f}%) in training set.")
        print("Skipping Classifier training -> Building Dummy-Classifier (always predicts 1.0).")

        clf = build_mlp(X_train.shape[1], [[2, 0.0, 0.0]], lr_schedule=LR, is_classifier=True)
        # Set final layer bias to +100.0 so sigmoid(100.0) == 1.0 everywhere
        final_layer = clf.layers[-1]
        dummy_weights = [np.zeros_like(final_layer.get_weights()[0]), np.array([100.0])]
        final_layer.set_weights(dummy_weights)

        clf.save(OUTPUT_DIR / "classifier_model.keras")
        history_clf = type("DummyHistory", (), {"history": {"bypass": [True]}})()

    else:
        # Compute balanced class weights and apply square-root damping to prevent
        # over-penalizing weak conducting currents near the blockade onset
        classes = np.unique(y_c_train.flatten())
        c_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_c_train.flatten())
        c_weights = np.sqrt(c_weights)
        class_weight_dict = dict(zip(classes.astype(int), c_weights))
        print(f"Classifier class weights (0=Blocked, 1=Conducting): {class_weight_dict}")

        steps_c = int(np.ceil(len(X_train) / BATCH_SIZE))
        lr_c = CosineDecay(initial_learning_rate=LR, decay_steps=steps_c * EPOCHS, alpha=0.01)

        clf = build_mlp(X_train.shape[1], CLASSIFIER_LAYERS, lr_c, is_classifier=True)
        history_clf = clf.fit(
            X_train, y_c_train,
            validation_data=(X_val, y_c_val),
            class_weight=class_weight_dict,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)],
            verbose=2,
        )
        clf.save(OUTPUT_DIR / "classifier_model.keras")

    # =========================================================================
    # 3. STAGE 2: Train Current Regressor (Conducting Regime Only)
    # =========================================================================
    print("\n--- 2. Training Current Regressor ---")
    if "bypass" in history_clf.history:
        # If Classifier is bypassed, train Regressor across the entire dataset
        cond_train = np.ones(len(y_c_train), dtype=bool)
        cond_val   = np.ones(len(y_c_val), dtype=bool)
    else:
        # Standard Hurdle setup: Train Regressor strictly on conducting samples
        cond_train = (y_c_train == 1.0).flatten()
        cond_val   = (y_c_val == 1.0).flatten()

    # Fit Target Scaler on the selected training samples
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_r_train = y_scaler.fit_transform(data["y_trans"][idx_train][cond_train])
    y_r_val   = y_scaler.transform(data["y_trans"][idx_val][cond_val])

    # Prepare sample weights from inverse KMC errors (clipped for gradient stability)
    if USE_ERROR_WEIGHTS:
        w_train = np.clip(data["weights"][idx_train][cond_train], *WEIGHT_CLIP_RANGE)
        w_val   = np.clip(data["weights"][idx_val][cond_val], *WEIGHT_CLIP_RANGE)
        # Re-normalize after clipping so the mean sample weight equals 1.0
        w_train /= np.mean(w_train)
        w_val   /= np.mean(w_val)
        print(f"Using clipped KMC error weights (Range: [{w_train.min():.2f}, {w_train.max():.2f}])")
    else:
        w_train = None
        w_val   = None
        print("Training regressor with uniform sample weights (USE_ERROR_WEIGHTS=False).")

    reg = build_mlp(X_train.shape[1], REGRESSOR_LAYERS, LR, is_classifier=False)
    reg_callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=45, restore_best_weights=True),
    ]
    history_reg = reg.fit(
        X_train[cond_train], y_r_train,
        sample_weight=w_train,
        validation_data=(X_val[cond_val], y_r_val, w_val) if w_val is not None else (X_val[cond_val], y_r_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=reg_callbacks,
        verbose=2,
    )
    reg.save(OUTPUT_DIR / "regressor_model.keras")

    # =========================================================================
    # 4. Save Artifacts, Feature Config, Histories & Hold-Out Test Set
    # =========================================================================
    with open(OUTPUT_DIR / "scalers.pkl", "wb") as f:
        pickle.dump({
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "scale_factor": SCALE_FACTOR,
            "blockade_threshold": BLOCKADE_THRESHOLD,
            "feature_config": {
                "diff_pairs": all_pairs,
                "sum_pairs": all_pairs,
                "prod_pairs": prod_pairs,
            },
        }, f)

    with open(OUTPUT_DIR / "training_history.pkl", "wb") as f:
        pickle.dump({
            "classifier": history_clf.history,
            "regressor": history_reg.history,
        }, f)

    # Save untouched test split in raw and transformed formats for Jupyter evaluation
    np.savez(
        OUTPUT_DIR / "test_data.npz",
        X_test_raw=data["X"][idx_test],
        X_test_scaled=X_test,
        y_test_class=data["y_class"][idx_test],
        y_test_raw=data["y_raw"][idx_test],
        y_test_trans=data["y_trans"][idx_test],
        weights_test=data["weights"][idx_test],
    )

    print(f"\nTraining complete! All models, scalers, and test data saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()