"""Bayesian MMM package."""

from .data_processing import DataConfig, DataProcessor, MMMDataset
from .model import BayesianMMM, ModelConfig, MMMResults
from .optimizer import BudgetOptimizer, OptimizerConfig, ChannelParams
from .diagnostics import check_convergence, out_of_sample_validation, generate_diagnostic_report
from . import visualization as viz

__all__ = [
    "DataConfig",
    "DataProcessor",
    "MMMDataset",
    "BayesianMMM",
    "ModelConfig",
    "MMMResults",
    "BudgetOptimizer",
    "OptimizerConfig",
    "ChannelParams",
    "check_convergence",
    "out_of_sample_validation",
    "generate_diagnostic_report",
    "viz",
]
