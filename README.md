# nanosurrogate

**Physics-Informed Deep Learning Surrogate for KMC Nanoparticle Networks & Gradient-Based In-Materio Computing**

`nanosurrogate` is a modular Python/TensorFlow package that replaces computationally expensive, stochastic Kinetic Monte Carlo (KMC) charge-transport simulations of disordered nanoparticle (NP) networks with a fast, end-to-end differentiable neural surrogate model. 

Designed around a $L \times L$ single-electron nanoparticle lattice with $N_e = 8$ electrodes ($7$ drive/gate inputs and $1$ grounded readout), the framework accurately resolves both the discrete **Coulomb blockade** regime and the strongly non-linear **tunneling transport** regime across eight orders of magnitude in current.

---

## Key Features

* **Two-Stage Hurdle Architecture:** Decouples the sharp physical phase transition at the Coulomb blockade boundary from continuous current regression to prevent neural Gibbs oscillations:
  * **Stage 1 (Blockade Classifier):** 4-layer Swish MLP with square-root damped class balancing predicting conduction probability $P(\vert{}I\vert{} > I_{\text{th}} \mid \vec{V})$.
  * **Stage 2 (Tunneling Regressor):** 6-layer Swish MLP trained exclusively on conducting states using inverse KMC error sample weights and Huber loss.
* **Physics-Informed Feature Expansion:** Automatically expands the $7$ raw electrode potentials into $70$ physical features capturing all $\binom{7}{2} = 21$ pairwise potential differences ($\Delta V_{ij} = V_i - V_j$), pairwise sums ($V_i + V_j$), and lateral cross-products ($V_i \cdot V_j$).
* **Multi-Scale Target Compression:** Uses an $\text{arcsinh}(I / s)$ transformation ($s = 1\,\text{pA} = 10^6\,\text{aA}$) to seamlessly bridge linear low-current fluctuations around zero and logarithmic high-current transport up to $10^8\,\text{aA}$.
* **Differentiable Hardware Twin (`DifferentiableNanoNode`):** Wraps the frozen classifier, regressor, and preprocessing scalers into a pure TensorFlow layer, enabling analytical gradients ($\partial I_{\text{out}} / \partial \vec{V}$) for instant gradient-descent tuning of control gate voltages.

---

## Repository Structure

```text
nanosurrogate/
├── data/
│   └── Nx=3_Ny=3_Ne=8_self_cap0.00.csv   # KMC simulation dataset (64,000 samples)
├── models/
│   └── hurdle_baseline/                  # Exported Keras models, scalers & test split
├── notebooks/                            # EDA, 1D I-V validation & In-Materio benchmarks
├── scripts/
│   └── train_hurdle.py                   # End-to-end training pipeline
├── src/
│   └── nanosurrogate/
│       ├── __init__.py
│       ├── env.py                        # GPU environment & memory growth setup
│       ├── data.py                       # Data loading, arcsinh scaling & feature expansion
│       └── models.py                     # MLP builders, HurdleSurrogate & DifferentiableNanoNode
└── pyproject.toml
```

---

## Installation

Clone the repository, create a virtual environment, and install the package in editable mode:

```bash
git clone https://github.com/JonasMnsing/nanosurrogate.git
cd nanosurrogate

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -e .
```

---

## Usage

### 1. Train the Two-Stage Hurdle Surrogate
Run the training script to perform the stratified $70/15/15$ split, fit the feature/target scalers, and train both network stages:

```bash
python scripts/train_hurdle.py
```
All artifacts (`classifier_model.keras`, `regressor_model.keras`, `scalers.pkl`, `training_history.pkl`, and `test_data.npz`) are exported to `models/hurdle_baseline/`.

### 2. Fast Forward Inference (`HurdleSurrogate`)
Use `HurdleSurrogate` for batch evaluation or 1D/2D $I$-$V$ characteristic sweeps in NumPy:

```python
import numpy as np
from itertools import combinations
from nanosurrogate.data import add_physics_features
from nanosurrogate.models import HurdleSurrogate

surrogate = HurdleSurrogate("models/hurdle_baseline")

# Example: Evaluate 100 random voltage configurations in [-0.03 V, +0.03 V]
V_raw = np.random.uniform(-0.03, 0.03, size=(100, 7))
all_pairs = list(combinations(range(7), 2))
X_feat = add_physics_features(V_raw, diff_pairs=all_pairs, sum_pairs=all_pairs, prod_pairs=all_pairs)

# Predict physical current in [aA] using smooth clamped soft gating
I_pred_aA = surrogate.predict(X_feat, prob_threshold=0.5, soft_gating=True)
```

### 3. Gradient-Based In-Materio Computing (`DifferentiableNanoNode`)
Use `DifferentiableNanoNode` with `tf.GradientTape` to optimize external control gate voltages for machine learning tasks (e.g., `make_moons` and `make_circles` classification):

```python
import tensorflow as tf
from nanosurrogate.models import DifferentiableNanoNode

node = DifferentiableNanoNode("models/hurdle_baseline")

# Differentiable forward pass from raw electrode voltages (batch_size, 7) [V] -> Current [pA]
V_input = tf.Variable(tf.zeros([1, 7], dtype=tf.float32))
with tf.GradientTape() as tape:
    I_pA = node(V_input)

dI_dV = tape.gradient(I_pA, V_input)  # Exact 7D differential conductance gradient!
```

---

## Benchmark Summary

| Task / Metric | Result | Notes |
| :--- | :--- | :--- |
| **Stage 1: Blockade Classifier** | $> 98\%$ Accuracy | ROC-AUC $> 0.998$ on $15\%$ hold-out test set |
| **Stage 2: Current Regressor** | $R^2 \approx 0.99$ | Evaluated across conducting states in $\text{arcsinh}$ space |
| **1D $I$-$V$ Sweep Validation** | NDR & Blockade resolved | Smoothly captures zero-current valleys and negative differential resistance |
| **In-Materio: Two Moons (2D)** | $98.5\%$ Accuracy | Single NP network ($3$ static control gates, $4$ data inputs) |
| **In-Materio: Concentric Circles** | $100.0\%$ Accuracy | Single NP network exploiting internal NDR potential well |