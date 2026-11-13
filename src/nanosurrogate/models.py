import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from typing import List, Tuple, Union

def build_mlp(input_dim: int, layer_info: List[Tuple[int, float, float]],
              lr_schedule: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule], is_classifier: bool = False) -> Model:
    """
    Constructs and compiles a Multi-Layer Perceptron (MLP / FNN) for either 
    Coulomb blockade classification or nonlinear current regression.

    Why an MLP with 'swish' activation:
    1. Dense layers compute linear combinations (w * V + b), which naturally align 
       with the diagonal hyperplanes of Coulomb diamonds in multi-gate voltage space.
    2. 'swish' (x * sigmoid(x)) is smooth and non-monotonic, avoiding dead neurons 
       (unlike ReLU) and providing continuous derivatives (dI/dV) for physical modeling.

    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., 7 raw electrode voltages + optional physics features).
    layer_info : List of Tuples (nodes, l2_reg, dropout_rate)
        Architecture definition per hidden layer. 
        Example: [[256, 1e-4, 0.0], [256, 1e-4, 0.1]] creates 2 hidden layers.
    lr_schedule : float or LearningRateSchedule
        Learning rate or dynamic schedule (e.g., CosineDecay) for the Adam optimizer.
    is_classifier : bool, default False
        If True: Builds the Gatekeeper Classifier (Sigmoid output, Binary Crossentropy loss).
        If False: Builds the Current Regressor (Linear output in arcsinh-space, Huber loss).

    Returns
    -------
    tf.keras.models.Model
        Compiled Keras model ready for training.
    """
    # The 'name' attribute to label for model.summary() and logs
    model_label = "Blockade_Classifier" if is_classifier else "Current_Regressor"
    model = Sequential(name=model_label)
    
    # Input Layer
    model.add(Input(shape=(input_dim,), name="voltage_inputs"))
    
    # Hidden Layers
    for i, (nodes, reg, drop) in enumerate(layer_info):
        model.add(Dense(nodes, kernel_regularizer=l2(reg), name=f"dense_{i+1}"))
        model.add(Activation('swish', name=f"swish_{i+1}"))
        if drop > 0.0:
            model.add(Dropout(drop, name=f"dropout_{i+1}"))
            
    # Output Layer & Compilation
    if is_classifier:
        # Sigmoid squashes output to a probability P(conducting) in [0.0, 1.0]
        model.add(Dense(1, activation='sigmoid', name="prob_conducting"))
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            loss='binary_crossentropy',
            metrics=[
                'accuracy', 
                tf.keras.metrics.AUC(name='auc') # AUC measures separation quality independent of threshold
            ]
        )
    else:
        # Linear output predicts the scaled, arcsinh-transformed current in [-1.0, 1.0]
        model.add(Dense(1, activation='linear', name="scaled_arcsinh_current"))
        model.compile(
            optimizer=Adam(learning_rate=lr_schedule),
            # Huber loss is quadratic for small errors and linear for large outliers,
            # preventing extreme KMC noise spikes from destabilizing gradients.
            loss='huber'
        )
        
    return model


class HurdleSurrogate:
    """
    Two-stage Hurdle Model wrapper combining the Blockade Classifier and Current Regressor 
    for end-to-end physical inference (Voltages [V] -> Physical Current [aA]).
    
    Mathematical Concept:
    Instead of forcing a single regressor to predict both flat 0.0 regions (Coulomb blockade)
    and exponential tunneling currents, inference is split into two stages:
      1. Classifier estimates conduction probability: P(conducting | V) in [0, 1]
      2. Regressor estimates conducting current magnitude: I_cond(V)
      3. Final current combines both via hard thresholding or smooth multiplicative gating.
    """
    def __init__(self, classifier: Model, regressor: Model, x_scaler, y_scaler, scale_factor: float = 0.16):
        """
        Parameters
        ----------
        classifier : Model
            Trained Keras classifier predicting conduction probability.
        regressor : Model
            Trained Keras regressor predicting scaled arcsinh current.
        x_scaler : sklearn.preprocessing.MinMaxScaler
            Fitted scaler for input voltage features.
        y_scaler : sklearn.preprocessing.MinMaxScaler
            Fitted scaler for arcsinh-transformed target currents.
        scale_factor : float, default 0.16
            The exact scale_factor (in aA) used during data preparation for arcsinh(I / s).
            Required to accurately invert the transformation via I = sinh(y_trans) * s.
        """
        self.classifier = classifier
        self.regressor = regressor
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.scale_factor = scale_factor

    def predict(self, X_raw: np.ndarray, prob_threshold: float = 0.5, soft_gating: bool = False, batch_size: int = 4096) -> np.ndarray:
        """
        Predicts physical electrical current (in original KMC units, e.g., aA) 
        from raw input voltages.

        Parameters
        ----------
        X_raw : np.ndarray of shape (n_samples, n_features)
            Unscaled input voltage matrix (must include physics features if trained with them).
        prob_threshold : float, default 0.5
            Decision boundary for the classifier (0.5 = 50% probability).
            Only used when soft_gating=False. If P(conducting) < prob_threshold, 
            the output current is forced strictly to 0.0.
        soft_gating : bool, default False
            If False (Hard Gating): Outputs exact 0.0 inside blockade. Best for reproducing 
                sharp KMC blockade plateaus.
            If True (Soft Gating): Multiplies continuous probability with predicted current 
                (I_final = P_cond * I_reg). Best when smooth gradients (dI/dV) are needed 
                for circuit simulations.
        batch_size : int, default 4096
            Batch size for GPU inference to prevent memory overflow on massive grids.

        Returns
        -------
        np.ndarray of shape (n_samples, 1)
            Predicted physical current in original units (aA).
        """
        # 1. Scale input voltages to [-1, 1]
        X_scaled = self.x_scaler.transform(X_raw)

        # 2. Stage 1: Predict Conduction Probability P in [0, 1]
        prob_cond = self.classifier.predict(X_scaled, verbose=0, batch_size=batch_size)

        # 3. Stage 2: Predict Current & Invert Transformations
        #    a) Neural net predicts in [-1, 1]
        y_scaled  = self.regressor.predict(X_scaled, verbose=0, batch_size=batch_size)
        #    b) Invert MinMaxScaler back to arcsinh space
        y_trans = self.y_scaler.inverse_transform(y_scaled)
        #    c) Invert arcsinh via sinh: I = sinh(y_trans) * scale_factor
        I_pred  = np.sinh(y_trans) * self.scale_factor

        # 4. Combine Classifier & Regressor
        if soft_gating:
            p_low  = max(0.05, prob_threshold - 0.15)
            p_high = min(0.95, prob_threshold + 0.25)
            gate   = np.clip((prob_cond - p_low) / (p_high - p_low), 0.0, 1.0)
            return gate * I_pred
        else:
            is_conducting = (prob_cond >= prob_threshold).astype(np.float32)
            return is_conducting * I_pred