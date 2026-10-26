"""Core AutoAIM modules.

Import modules directly to avoid loading heavy dependencies:
    from app.core.data_manager import DataManager
    from app.core.trainer import ModelTrainer
    from app.core.nn_builder import NeuralNetworkBuilder  # loads torch
"""

# Deliberately empty: no imports at package level.
# This prevents torch from loading when importing unrelated modules.
# All code imports directly from submodules.
