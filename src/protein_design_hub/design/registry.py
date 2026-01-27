"""Designer registry (factory pattern) for sequence/backbone design tools."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from protein_design_hub.core.config import Settings
from protein_design_hub.core.exceptions import PredictorNotFoundError
from protein_design_hub.design.base import BaseDesigner


class DesignerRegistry:
    _designers: Dict[str, Type[BaseDesigner]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(designer_class: Type[BaseDesigner]):
            cls._designers[name.lower()] = designer_class
            return designer_class

        return decorator

    @classmethod
    def get(cls, name: str, settings: Optional[Settings] = None) -> BaseDesigner:
        name_lower = name.lower().replace("-", "_")
        aliases = {
            "proteinmpnn": "proteinmpnn",
            "mpnn": "proteinmpnn",
        }
        resolved = aliases.get(name_lower, name_lower)
        if resolved not in cls._designers:
            available = ", ".join(sorted(cls._designers.keys()))
            raise PredictorNotFoundError(f"Unknown designer: {name}. Available: {available}")
        return cls._designers[resolved](settings)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._designers.keys())


def get_designer(name: str, settings: Optional[Settings] = None) -> BaseDesigner:
    return DesignerRegistry.get(name, settings)


def _register_designers() -> None:
    try:
        from protein_design_hub.design.proteinmpnn import designer as _  # noqa: F401
    except Exception:
        pass


_register_designers()
