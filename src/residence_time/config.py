import numpy as np

# Feature extension time
DEFAULT_TIME_ENCODING_CONFIG = {
    "enabled": False,
    "use_cyclical": False,
    "use_rbf_seasonal": False,
    "use_rbf_absolute": False,
    "n_rbf_seasonal": 12,
    "n_rbf_absolute": 24,
    "gamma_seasonal": 100.0,
    "gamma_absolute": 5.0,
    "t_min": 1980.0,
    "t_max": 2025.0
}

# Feature extension tropopoause
USE_TROPOPAUSE_FEATURES = True

# Model settings
INPUT_DIM = 7
HIDDEN_DIM = 32
HIDDEN_LAYERS = 2
INCLUDE_D = False
DEVICE = 'cuda'

# Training
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 10

# Physics informed neural network constraint
PHYSICS_CONSTRAINT = "diffusivity"  # Options: "harmonic", "diffusivity"

# Harmonic oscillator settings
OMEGA = 2 * np.pi  # For harmonic oscillator (1 cycle/year)
MEAN_TAU = 1.5
