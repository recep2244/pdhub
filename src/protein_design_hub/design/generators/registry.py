"""Registry for backbone generators."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from protein_design_hub.core.config import Settings
from protein_design_hub.core.exceptions import PredictorNotFoundError
from protein_design_hub.design.generators.base import BaseBackboneGenerator


class GeneratorRegistry:
    _generators: Dict[str, Type[BaseBackboneGenerator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(gen_class: Type[BaseBackboneGenerator]):
            cls._generators[name.lower()] = gen_class
            return gen_class

        return decorator

    @classmethod
    def get(cls, name: str, settings: Optional[Settings] = None) -> BaseBackboneGenerator:
        name_lower = name.lower().replace("-", "_")
        aliases = {
            "rfdiffusion": "rfdiffusion",
            "rf_diffusion": "rfdiffusion",
        }
        resolved = aliases.get(name_lower, name_lower)
        if resolved not in cls._generators:
            available = ", ".join(sorted(cls._generators.keys()))
            raise PredictorNotFoundError(f"Unknown generator: {name}. Available: {available}")
        return cls._generators[resolved](settings)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._generators.keys())


def get_generator(name: str, settings: Optional[Settings] = None) -> BaseBackboneGenerator:
    return GeneratorRegistry.get(name, settings)


def _register_generators() -> None:
    try:
        from protein_design_hub.design.rfdiffusion import generator as _  # noqa: F401
    except Exception:
        pass


_register_generators()
