import numpy as np
import tensorflow as tf
import pickle

from typing import List, Tuple, Union
from itertools import combinations
from pathlib import Path

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

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

class DifferentiableNanoNode(tf.keras.layers.Layer):
    """
    A frozen, fully differentiable TensorFlow/Keras layer wrapping the trained 
    Two-Stage Hurdle Surrogate Model for gradient-based In-Materio Computing.

    Purpose & Physical Background
    -----------------------------
    Standard surrogate evaluation (`HurdleSurrogate`) uses NumPy and scikit-learn 
    scalers, which breaks TensorFlow's automatic differentiation tape (`tf.GradientTape`).
    By re-implementing the entire preprocessing pipeline (pairwise voltage differences, 
    pairwise sums, voltage products, and MinMax scaling) and the inverse `arcsinh` 
    transformation in pure TensorFlow operations, this layer allows gradients to 
    flow seamlessly from the predicted physical output current $I_{\\mathrm{out}}$ 
    back to the input electrode voltages $\\vec{V} = (V_0, \\dots, V_6)$.

    During optimization, all internal neural network weights of the classifier and 
    regressor remain strictly frozen (`trainable = False`). Instead, the external 
    control voltages applied to the nanoparticle network are optimized via gradient 
    descent to program the physical device for a target computational task.

    Attributes
    ----------
    clf : tf.keras.Model
        Frozen Stage-1 Coulomb blockade classifier predicting $P(\\mathrm{conducting} \\mid \\vec{V})$.
    reg : tf.keras.Model
        Frozen Stage-2 tunneling current regressor predicting scaled $\\mathrm{arcsinh}(I / s)$.
    scale_factor : tf.Tensor
        Transition current scale $s$ [aA] used in the inverse $\\sinh$ transformation.
    diff_pairs : list of tuple[int, int]
        Electrode index pairs $(i, j)$ for pairwise voltage differences $V_i - V_j$.
    sum_pairs : list of tuple[int, int]
        Electrode index pairs $(i, j)$ for pairwise voltage sums $V_i + V_j$.
    prod_pairs : list of tuple[int, int]
        Electrode index pairs $(i, j)$ for non-linear voltage products $V_i \\cdot V_j$.
    x_scale, x_min : tf.Tensor
        Extracted affine transformation constants from `MinMaxScaler` for input features.
    y_scale, y_min : tf.Tensor
        Extracted affine transformation constants from `MinMaxScaler` for target currents.
    """

    def __init__(self, model_dir: str | Path, **kwargs) -> None:
        """
        Initializes the differentiable hardware node from exported training artifacts.

        Parameters
        ----------
        model_dir : str or Path
            Path to the directory containing `classifier_model.keras`, 
            `regressor_model.keras`, and `scalers.pkl`.
        **kwargs : dict
            Additional keyword arguments passed to `tf.keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        model_dir = Path(model_dir)

        # Load both stages without compilation (only forward inference & input gradients needed)
        self.clf = load_model(model_dir / "classifier_model.keras", compile=False)
        self.reg = load_model(model_dir / "regressor_model.keras", compile=False)

        # Freeze internal neural network weights so only external voltages are optimized
        self.clf.trainable = False
        self.reg.trainable = False

        with open(model_dir / "scalers.pkl", "rb") as f:
            meta = pickle.load(f)

        self.scale_factor = tf.constant(meta["scale_factor"], dtype=tf.float32)

        # Load feature engineering topology (fallback to all 21 pairs for 7 electrodes)
        default_pairs = list(combinations(range(7), 2))
        f_cfg = meta.get(
            "feature_config",
            {
                "diff_pairs": default_pairs,
                "sum_pairs": default_pairs,
                "prod_pairs": default_pairs,
            },
        )
        self.diff_pairs = f_cfg["diff_pairs"]
        self.sum_pairs  = f_cfg["sum_pairs"]
        self.prod_pairs = f_cfg["prod_pairs"]

        # Extract scikit-learn MinMaxScaler parameters as TensorFlow constants:
        # X_scaled = X_feat * x_scale + x_min
        x_sc = meta["x_scaler"]
        y_sc = meta["y_scaler"]
        self.x_scale = tf.constant(x_sc.scale_, dtype=tf.float32)
        self.x_min   = tf.constant(x_sc.min_, dtype=tf.float32)
        self.y_scale = tf.constant(y_sc.scale_[0], dtype=tf.float32)
        self.y_min   = tf.constant(y_sc.min_[0], dtype=tf.float32)

    def call(self, V_raw: tf.Tensor) -> tf.Tensor:
        """
        Performs a differentiable forward pass from raw electrode voltages to output current.

        Note on Smooth Gating
        ---------------------
        Unlike hard thresholding (which has a zero derivative everywhere and blocks 
        gradient flow), this layer uses continuous expectation gating:
        $I_{\\mathrm{out}} = P(\\mathrm{conducting} \\mid \\vec{V}) \\cdot I_{\\mathrm{reg}}(\\vec{V})$.
        This ensures non-vanishing gradients even when an initial voltage configuration 
        starts deep inside the Coulomb blockade regime.

        Parameters
        ----------
        V_raw : tf.Tensor
            Input tensor of shape `(batch_size, n_electrodes)` containing raw electrode 
            potentials in Volts (typically bounded within `[-0.03, +0.03]` V).

        Returns
        -------
        I_pA : tf.Tensor
            Predicted physical output current tensor of shape `(batch_size, 1)` in 
            picoamperes [pA] ($1\\,\\mathrm{pA} = 10^6\\,\\mathrm{aA}$).
        """
        # 1. Differentiable physics feature expansion
        features = [V_raw]
        for i, j in self.diff_pairs:
            features.append(V_raw[:, i : i + 1] - V_raw[:, j : j + 1])
        for i, j in self.sum_pairs:
            features.append(V_raw[:, i : i + 1] + V_raw[:, j : j + 1])
        for i, j in self.prod_pairs:
            features.append(V_raw[:, i : i + 1] * V_raw[:, j : j + 1])

        X_feat = tf.concat(features, axis=1)

        # 2. Scale features to [-1, 1] using training set statistics
        X_scaled = X_feat * self.x_scale + self.x_min

        # 3. Evaluate frozen Classifier and Regressor networks
        prob_cond = self.clf(X_scaled, training=False)
        y_scaled  = self.reg(X_scaled, training=False)

        # 4. Inverse scaling and inverse arcsinh transformation back to physical current
        y_trans = (y_scaled - self.y_min) / self.y_scale
        I_aA    = tf.math.sinh(y_trans) * self.scale_factor

        # 5. Return smoothly gated expectation current in picoamperes [pA]
        return (prob_cond * I_aA) / 1e6