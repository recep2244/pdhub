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
        return "qwen2.5:14b"


@dataclass(frozen=True, eq=True)
class LLMAgent:
    """An LLM-powered scientist agent.

    Attributes:
        title: Display name (e.g. "Principal Investigator").
        expertise: Area of expertise used in the system prompt.
        goal: What this agent aims to achieve.
        role: Operational role description.
        model: LLM model identifier.  Defaults to ``settings.llm.model``
               (e.g. ``qwen2.5:14b`` for Ollama, ``gpt-4o`` for OpenAI).
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
        """System prompt that defines the agent's persona.

        Structured for deep scientific analysis with 14B-70B local models
        (qwen2.5:14b, deepseek-r1:14b) or cloud APIs.
        """
        return (
            f"# {self.title}\n\n"
            f"## Expertise\n{self.expertise}\n\n"
            f"## Goal\n{self.goal}\n\n"
            f"## Domain Knowledge & Analysis Framework\n{self.role}\n\n"
            "## Response standards\n"
            "- Reference specific numeric thresholds when evaluating metrics\n"
            "- Structure your response: Assessment → Evidence → Recommendation\n"
            "- Flag uncertainty explicitly: distinguish low-confidence claims\n"
            "- Cite the relevant tool or method for each recommendation\n"
            "- Be actionable: specify the next concrete experiment or calculation\n\n"
            "## Universal thresholds\n"
            "pLDDT: >90=excellent, 70-90=confident, 50-70=low, <50=disordered\n"
            "TM-score: >0.7=high similarity, >0.5=same fold, <0.5=different fold\n"
            "RMSD: <1Å=near-native, <2Å=good, >3Å=significant deviation\n"
            "MolProbity clash: <10=excellent, <20=good, >40=poor\n"
            "ModFOLD9 p-value: <0.001=confident, <0.01=good, >0.1=unreliable\n"
            "DockQ: >0.8=high, 0.49-0.8=medium, 0.23-0.49=acceptable\n"
            "GDT-TS: >80=top tier, >60=good, >40=moderate, <40=poor\n\n"
            "## Available tools (recommend by name when applicable)\n"
            "Prediction: IntFOLD7, MultiFOLD2, ColabFold, ESMFold, Chai-1, Boltz-2\n"
            "QA: ModFOLD9, ModFOLDdock2, MolProbity, ProQ3D, VoroMQA\n"
            "Function: FunFOLD5, P2Rank, DISOclust, DomFOLD\n"
            "Refinement: ReFOLD3, GalaxyRefine, AMBER, Rosetta FastRelax\n"
            "Design: ProteinMPNN, LigandMPNN, SolubleMPNN, RFdiffusion\n"
            "Biophysics: FoldX, Rosetta ddG, TANGO, CamSol, NetMHCpan"
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
