"""
Constants used throughout the most_queue package.
"""

# Default number of states for simulation
DEFAULT_NUM_STATES: int = 100000

# Default number of jobs for simulation
DEFAULT_NUM_JOBS: int = 1000000

# Default number of probabilities to calculate
DEFAULT_P_NUM: int = 1000

# Numerical tolerances
DEFAULT_TOLERANCE: float = 1e-12
LSTSQ_RCOND: float = 1e-8  # rcond parameter for numpy.linalg.lstsq

# Maximum iterations for fitting algorithms
MAX_FIT_ITERATIONS: int = 10000

# Default number of dots for approximation
DEFAULT_DOTS_NUM: int = 10000
