"""
Advanced Stock Prediction System

A sophisticated machine learning-based stock prediction system using 
extended sequence encoding and multiple parameter optimization techniques.
"""

__version__ = "1.0.0"
__author__ = "SaiChen22"
__email__ = "your-email@domain.com"

# Import main classes and functions
from .encoders import (
    SimpleEncoder,
    ExtendedEncoder,
    create_test_sequences
)

from .optimizers import (
    BayesianOptimizer,
    GeneticOptimizer,
    RandomSearchOptimizer,
    compare_optimizers
)

# Import main prediction functions
from .stock_prediction_extended_encoding import (
    extended_prediction_score,
    original_prediction_score,
    get_training_test_data
)

# Import ML optimization
from .ml_parameter_optimization import (
    MLParameterOptimizer,
    BayesianOptimizer as MLBayesianOptimizer,
    compare_optimization_methods
)

__all__ = [
    'SimpleEncoder',
    'ExtendedEncoder', 
    'create_test_sequences',
    'BayesianOptimizer',
    'GeneticOptimizer',
    'RandomSearchOptimizer',
    'compare_optimizers',
    'extended_prediction_score',
    'original_prediction_score',
    'get_training_test_data',
    'MLParameterOptimizer',
    'MLBayesianOptimizer',
    'compare_optimization_methods'
]