import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_mlp(input_dim: int, layer_info: list, lr_schedule, is_classifier: bool = False) -> Model:
    """Builds either the blockade classifier or the current regressor."""
    model = Sequential(name="Blockade_Classifier" if is_classifier else "Current_Regressor")
    model.add(Input(shape=(input_dim,)))
    
    for nodes, reg, drop in layer_info:
        model.add(Dense(nodes, kernel_regularizer=l2(reg)))
        model.add(Activation('swish'))
        if drop > 0:
            model.add(Dropout(drop))
            
    if is_classifier:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
    else:
        model.add(Dense(1, activation='linear'))
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='huber'
        )
    return model

class HurdleSurrogate:
    """Combines Classifier and Regressor for fast physical inference."""
    def __init__(self, classifier: Model, regressor: Model, x_scaler, y_scaler, scale_factor: float = 0.16):
        self.classifier = classifier
        self.regressor = regressor
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.scale_factor = scale_factor

    def predict(self, X_raw: np.ndarray, threshold: float = 0.5, soft_gating: bool = False) -> np.ndarray:
        """
        Predicts physical current in original units.
        soft_gating=True multiplies probability with current (smooth derivatives!).
        soft_gating=False sets current strictly to 0.0 if P(conducting) < threshold.
        """
        X_scaled = self.x_scaler.transform(X_raw)
        
        # 1. Predict Blockade Probability
        prob_conducting = self.classifier.predict(X_scaled, verbose=0)
        
        # 2. Predict Current (in scaled arcsinh space)
        y_scaled_pred = self.regressor.predict(X_scaled, verbose=0)
        y_trans_pred = self.y_scaler.inverse_transform(y_scaled_pred)
        current_pred = np.sinh(y_trans_pred) * self.scale_factor
        
        # 3. Combine
        if soft_gating:
            return prob_conducting * current_pred
        else:
            is_conducting = (prob_conducting >= threshold).astype(np.float32)
            return is_conducting * current_pred