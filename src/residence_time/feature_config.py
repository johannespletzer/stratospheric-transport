from enum import IntEnum


class FeatureIndex(IntEnum):
    """Column indices for base and extended features."""

    LAT = 0
    ALT = 1
    TIME = 2
    SOURCE = 3
    GAMMA = 4
    TAU_FLAG = 5
    TP_TROPO = 6
    TP_SH = 7
    TP_NH = 8
    TP_FLAG = 9

# Number of core input features without tropopause data
N_BASE_FEATURES = FeatureIndex.TAU_FLAG + 1

# Number of additional columns added when tropopause features are appended
N_TP_FEATURES = FeatureIndex.TP_FLAG - FeatureIndex.TAU_FLAG
