"""Bayesian MMM package.

Keep package imports lazy so importing a specific submodule (e.g.
``src.data_processing``) does not eagerly import optimizer/model stacks.
This avoids unrelated import-time failures and circular import edge cases.
"""

from importlib import import_module

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


_LAZY_IMPORTS = {
    "DataConfig": ("src.data_processing", "DataConfig"),
    "DataProcessor": ("src.data_processing", "DataProcessor"),
    "MMMDataset": ("src.data_processing", "MMMDataset"),
    "BayesianMMM": ("src.model", "BayesianMMM"),
    "ModelConfig": ("src.model", "ModelConfig"),
    "MMMResults": ("src.model", "MMMResults"),
    "BudgetOptimizer": ("src.optimizer", "BudgetOptimizer"),
    "OptimizerConfig": ("src.optimizer", "OptimizerConfig"),
    "ChannelParams": ("src.optimizer", "ChannelParams"),
    "check_convergence": ("src.diagnostics", "check_convergence"),
    "out_of_sample_validation": ("src.diagnostics", "out_of_sample_validation"),
    "generate_diagnostic_report": ("src.diagnostics", "generate_diagnostic_report"),
    "viz": ("src.visualization", None),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'src' has no attribute '{name}'")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value
