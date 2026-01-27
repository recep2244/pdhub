"""Predictor registry for factory pattern."""

from typing import Dict, List, Optional, Type

from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.core.types import PredictorType
from protein_design_hub.core.config import Settings
from protein_design_hub.core.exceptions import PredictorNotFoundError


class PredictorRegistry:
    """Registry for managing available predictors."""

    _predictors: Dict[str, Type[BasePredictor]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a predictor class.

        Usage:
            @PredictorRegistry.register("chai1")
            class Chai1Predictor(BasePredictor):
                ...
        """
        def decorator(predictor_class: Type[BasePredictor]):
            cls._predictors[name.lower()] = predictor_class
            return predictor_class
        return decorator

    @classmethod
    def get(cls, name: str, settings: Optional[Settings] = None) -> BasePredictor:
        """
        Get a predictor instance by name.

        Args:
            name: Predictor name (case-insensitive).
            settings: Optional settings to pass to the predictor.

        Returns:
            Predictor instance.

        Raises:
            PredictorNotFoundError: If predictor is not registered.
        """
        name_lower = name.lower()

        # Handle aliases
        aliases = {
            "colabfold": "colabfold",
            "localcolabfold": "colabfold",
            "alphafold2": "colabfold",
            "af2": "colabfold",
            "chai": "chai1",
            "chai-1": "chai1",
            "boltz": "boltz2",
            "boltz-2": "boltz2",
            "esm": "esmfold",
            "esm-fold": "esmfold",
            "esmfold-api": "esmfold_api",
            "esm-api": "esmfold_api",
        }

        resolved_name = aliases.get(name_lower, name_lower)

        if resolved_name not in cls._predictors:
            available = ", ".join(cls._predictors.keys())
            raise PredictorNotFoundError(
                f"Unknown predictor: {name}. Available: {available}"
            )

        return cls._predictors[resolved_name](settings)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered predictor names."""
        return list(cls._predictors.keys())

    @classmethod
    def get_all(cls, settings: Optional[Settings] = None) -> Dict[str, BasePredictor]:
        """
        Get all registered predictors.

        Args:
            settings: Optional settings to pass to predictors.

        Returns:
            Dictionary of predictor name to instance.
        """
        return {name: pred_cls(settings) for name, pred_cls in cls._predictors.items()}

    @classmethod
    def get_installed(cls, settings: Optional[Settings] = None) -> List[BasePredictor]:
        """
        Get all installed predictors.

        Args:
            settings: Optional settings to pass to predictors.

        Returns:
            List of installed predictor instances.
        """
        installed = []
        for name, pred_cls in cls._predictors.items():
            predictor = pred_cls(settings)
            if predictor.installer.is_installed():
                installed.append(predictor)
        return installed


def get_predictor(name: str, settings: Optional[Settings] = None) -> BasePredictor:
    """
    Convenience function to get a predictor by name.

    Args:
        name: Predictor name.
        settings: Optional settings.

    Returns:
        Predictor instance.
    """
    return PredictorRegistry.get(name, settings)


# Import predictors to trigger registration
def _register_predictors():
    """Import all predictor modules to register them."""
    try:
        from protein_design_hub.predictors.colabfold import predictor as _  # noqa: F401
    except ImportError:
        pass

    try:
        from protein_design_hub.predictors.chai1 import predictor as _  # noqa: F401
    except ImportError:
        pass

    try:
        from protein_design_hub.predictors.boltz2 import predictor as _  # noqa: F401
    except ImportError:
        pass

    try:
        from protein_design_hub.predictors.esmfold import predictor as _  # noqa: F401
    except ImportError:
        pass


_register_predictors()
