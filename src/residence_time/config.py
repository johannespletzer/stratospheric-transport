import numpy as np

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
    "t_max": 2025.0,
    "time_col": 2
}

PHYSICS_CONSTRAINT = "diffusivity"  # Options: "harmonic", "diffusivity"
OMEGA = 2 * np.pi  # For harmonic oscillator (1 cycle/year)

MEAN_TAU = 1.5
USE_TROPOPAUSE_FEATURES = False

LEARNING_RATE = 1e-4
