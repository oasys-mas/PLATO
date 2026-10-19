"""
Execution package for evaluating different RL algorithms.

This package provides a clean, object-oriented interface for evaluating
trained models across different algorithms (PLATO, DICG, DGN, MOHITO).

Main classes:
- EvaluationManager: Orchestrates evaluation across algorithms
- BaseEvaluator: Base class for all evaluators
- PLATOEvaluator: Evaluator for PLATO algorithm
- DICGEvaluator: Evaluator for DICG algorithm  
- DGNEvaluator: Evaluator for DGN algorithm
- MOHITOEvaluator: Evaluator for MOHITO algorithm

Usage:
    from execution import EvaluationManager
    
    manager = EvaluationManager()
    results = manager.evaluate_all_experiments()
"""

from .naive_baseline_evaluator import NaiveBaselineEvaluator
from .base_evaluator import BaseEvaluator
from .PLATO_evaluator import PLATOEvaluator
from .DICG_evaluator import DICGEvaluator
from .DGN_evaluator import DGNEvaluator
from .DEEPSET_evaluator import DEEPSETEvaluator
from .mohito_evaluator import MOHITOEvaluator

__version__ = "1.0.0"
__author__ = "Ali"

__all__ = [
    "EvaluationManager",
    "BaseEvaluator", 
    "PLATOEvaluator",
    "DICGEvaluator",
    "DGNEvaluator", 
    "MOHITOEvaluator",
    "NaiveBaselineEvaluator"
]