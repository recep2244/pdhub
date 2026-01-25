"""Predictors for protein structure prediction."""

from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.predictors.registry import PredictorRegistry, get_predictor

__all__ = [
    "BasePredictor",
    "PredictorRegistry",
    "get_predictor",
]
