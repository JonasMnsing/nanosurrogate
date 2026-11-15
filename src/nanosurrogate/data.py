import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def load_and_prepare_data(filepath: str, scale_factor: float = 0.16, threshold: float = 0.16) -> Dict[str, np.ndarray]:
    """
    Loads KMC data, separates the Coulomb blockade regime, applies physical scaling, 
    and computes error weights for neural network training.

    Parameters
    ----------
    filepath : str
        Path to the raw KMC dataset (.csv).
    scale_factor : float, default 0.16
        Transition point for the arcsinh transformation. 
        Currents |I| < scale_factor are treated linearly.
        Currents |I| > scale_factor are compressed logarithmically.
        Default 0.16 aA corresponds to approx. 1 electron per second.
    threshold : float, default 0.16
        Current threshold below which the system is considered in Coulomb Blockade (0.0).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing features, targets (raw and transformed), and sample weights.
    """
    df = pd.read_csv(filepath)
    
    # ---------------------------------------------------------
    # 1. Identify Coulomb Blockade (Classifier Target)
    # ---------------------------------------------------------
    # If the current is below 1 e/s OR the KMC error is exactly 0.0 (no jumps),
    # the system is firmly in the Coulomb blockade regime.
    is_blocked = (df['Observable'].abs() < threshold) | (df['Error'] == 0.0)
    y_class = (~is_blocked).astype(np.float32).values.reshape(-1, 1) # 1.0 = conducting, 0.0 = blocked
    
    # ---------------------------------------------------------
    # 2. Physics-Informed Features
    # ---------------------------------------------------------
    # Extract raw voltages (first 7 columns)
    X = df.iloc[:, :7].values.astype(np.float32)
    
    # ---------------------------------------------------------
    # 3. Target & Error Transformation (arcsinh space)
    # ---------------------------------------------------------
    y_raw = df['Observable'].values.reshape(-1, 1)
    y_err = df['Error'].values.reshape(-1, 1)
    
    # arcsinh handles values spanning multiple orders of magnitude AND negative/zero values.
    # We apply this to ALL values here for vectorization speed, but the regressor 
    # will only see the values where y_class == 1.0 later.
    y_trans = np.arcsinh(y_raw / scale_factor).astype(np.float32)
    
    # Error propagation through arcsinh: d/dy arcsinh(y/s) = 1 / sqrt(y^2 + s^2)
    sigma_trans = y_err / np.sqrt(y_raw**2 + scale_factor**2)
    
    # ---------------------------------------------------------
    # 4. Compute Inverse-Error Weights
    # ---------------------------------------------------------
    # We assign weights ONLY to conducting samples (blocked samples aren't regressed anyway).
    weights = np.ones_like(y_raw, dtype=np.float32)
    cond_mask = (y_class.flatten() == 1.0)
    
    # Prevent division by zero if a conducting sample has an unusually tiny error
    safe_sigma = np.maximum(sigma_trans[cond_mask], 1e-4)
    raw_weights = 1.0 / safe_sigma
    
    # Normalize weights so their mean is 1.0 (keeps learning rates stable)
    weights[cond_mask] = raw_weights / np.mean(raw_weights)
    
    return {
        "X": X,
        "y_class": y_class,
        "y_trans": y_trans,
        "y_raw": y_raw,
        "weights": weights.flatten()
    }

def add_physics_features(X: np.ndarray, diff_pairs: Optional[List[Tuple[int, int]]] = None, sum_pairs: Optional[List[Tuple[int, int]]] = None, prod_pairs: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    """
    Dynamically appends physical interaction features to the raw voltage vector.
    
    Parameters
    ----------
    X_raw : np.ndarray
        The base feature matrix (e.g., raw electrode voltages).
    diff_pairs : List of Tuples, optional
        Pairs of column indices (i, j) to compute differences (V_i - V_j).
    prod_pairs : List of Tuples, optional
        Pairs of column indices (i, j) to compute products (V_i * V_j).
        
    Returns
    -------
    np.ndarray
        Expanded feature matrix.
    """
    features = [X]

    if diff_pairs:
        for i, j in diff_pairs:
            features.append((X[:, i] - X[:, j]).reshape(-1, 1))

    if sum_pairs:
        for i, j in sum_pairs:
            features.append((X[:, i] + X[:, j]).reshape(-1, 1))

    if prod_pairs:
        for i, j in prod_pairs:
            features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    return np.hstack(features)