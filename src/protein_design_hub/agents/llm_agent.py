"""LLM-powered scientist agent, inspired by Virtual Lab (Zou group).

Each agent has a title, expertise, goal, and role that define its
system prompt.  Agents can participate in team meetings (multi-agent
round-robin discussion) and individual meetings (agent + critic loop).

The LLM backend is configured via ``Settings.llm`` (see ``core/config.py``).
By default it points to Ollama on localhost so everything runs locally
without any cloud API key.

Reference: https://github.com/zou-group/virtual-lab
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


def _default_model() -> str:
    """Return the default model from settings (lazy, avoids import cycle)."""
    try:
        from protein_design_hub.core.config import get_settings
        return get_settings().llm.resolve().model
    except Exception:
        return "llama3.2:latest"


@dataclass(frozen=True, eq=True)
class LLMAgent:
    """An LLM-powered scientist agent.

    Attributes:
        title: Display name (e.g. "Principal Investigator").
        expertise: Area of expertise used in the system prompt.
        goal: What this agent aims to achieve.
        role: Operational role description.
        model: LLM model identifier.  Defaults to ``settings.llm.model``
               (e.g. ``llama3.2`` for Ollama, ``gpt-4o`` for OpenAI).
    """

    title: str
    expertise: str
    goal: str
    role: str
    model: str = ""  # empty ⇒ use config default at call time

    # -----------------------------------------------------------------
    # Prompt helpers
    # -----------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """System prompt that defines the agent's persona."""
        return (
            f"You are a {self.title}. "
            f"Your expertise is in {self.expertise}. "
            f"Your goal is to {self.goal}. "
            f"Your role is to {self.role}. "
            "Always ground your advice in specific metrics, thresholds, and "
            "evidence. When discussing protein structures, reference concrete "
            "scores (pLDDT > 90 = excellent, TM-score > 0.5 = same fold, "
            "RMSD < 2 Å = near-native, clash score < 10 = well-packed, "
            "ModFOLD p-value < 0.001 = high confidence, ModFOLDdock2 "
            "QSCORE for interface quality, GDT-TS > 70 = excellent match). "
            "You are part of a protein design system co-developed with the "
            "McGuffin Lab (University of Reading). The lab's tools include: "
            "IntFOLD7 (integrated prediction+QA+function), MultiFOLD2 "
            "(tertiary+quaternary with stoichiometry; top CASP16 server), "
            "ModFOLD9 (model QA with p-values), ModFOLDdock2 (interface QA; "
            "CASP16 #1), ReFOLD3 (quality-guided refinement), FunFOLD5 "
            "(binding site prediction), DISOclust (disorder), DomFOLD "
            "(domain boundaries). Reference these tools when relevant. "
            "Be concise, actionable, and scientifically rigorous."
        )

    @property
    def system_message(self) -> Dict[str, str]:
        """OpenAI-style system message dict."""
        return {"role": "system", "content": self.system_prompt}

    @property
    def resolved_model(self) -> str:
        """Return the model to use (agent override or global config)."""
        return self.model if self.model else _default_model()

    # -----------------------------------------------------------------
    # Dunder helpers
    # -----------------------------------------------------------------

    def __str__(self) -> str:
        return self.title

    def __repr__(self) -> str:
        return f"LLMAgent(title={self.title!r}, model={self.resolved_model!r})"

    def __hash__(self) -> int:
        return hash(self.title)
