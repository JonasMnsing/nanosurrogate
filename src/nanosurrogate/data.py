import numpy as np
import pandas as pd
from typing import Dict

def load_and_prepare_data(filepath: str, scale_factor: float = 0.16, threshold: float = 0.16) -> Dict[str, np.ndarray]:
    """Loads data, separates blockade regime, and computes error weights."""
    df = pd.read_csv(filepath)
    
    # 1. Identify Coulomb Blockade
    is_blocked = (df['Observable'].abs() < threshold) | (df['Error'] == 0.0)
    y_class = (~is_blocked).astype(np.float32).values.reshape(-1, 1) # 1.0 = conducting, 0.0 = blocked
    
    # 2. Physics-Informed Features
    V = df.iloc[:, :7].values
    v4_minus_v6 = (V[:, 4] - V[:, 6]).reshape(-1, 1)
    v4_v6_inter = (V[:, 4] * V[:, 6]).reshape(-1, 1)
    X = np.hstack([V, v4_minus_v6, v4_v6_inter]).astype(np.float32)
    
    # 3. Target & Error Transformation (arcsinh space)
    y_raw = df['Observable'].values.reshape(-1, 1)
    y_err = df['Error'].values.reshape(-1, 1)
    
    y_trans = np.arcsinh(y_raw / scale_factor).astype(np.float32)
    
    # Error propagation for arcsinh(y / s): d/dy arcsinh(y/s) = 1 / sqrt(y^2 + s^2)
    sigma_trans = y_err / np.sqrt(y_raw**2 + scale_factor**2)
    
    # Compute inverse-error weights for conducting samples (normalized to mean=1)
    weights = np.ones_like(y_raw, dtype=np.float32)
    cond_mask = (y_class.flatten() == 1.0)
    
    # Avoid division by zero, then invert
    safe_sigma = np.maximum(sigma_trans[cond_mask], 1e-4)
    raw_weights = 1.0 / safe_sigma
    weights[cond_mask] = raw_weights / np.mean(raw_weights)
    
    return {
        "X": X,
        "y_class": y_class,
        "y_trans": y_trans,
        "y_raw": y_raw,
        "weights": weights.flatten()
    }