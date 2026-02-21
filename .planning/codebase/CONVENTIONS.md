# Coding Conventions

**Analysis Date:** 2026-02-21

## Naming Patterns

**Files:**
- Module files: lowercase with underscores (e.g., `llm_agent.py`, `meeting.py`, `orchestrator.py`)
- Dataclass/model files: `types.py` (centralized type definitions), `config.py` (configuration)
- Agent implementations: `*_agent.py` (e.g., `prediction_agent.py`, `input_agent.py`)
- Test files: `test_*.py` following module name patterns

**Functions:**
- snake_case for all function names: `_get_llm_client()`, `run_meeting()`, `calculate_biophysical_metrics()`
- Private/internal functions: prefix with underscore (e.g., `_strip_think_blocks()`, `_parse_verdict_from_summary()`)
- Public API functions: no underscore prefix
- Helper functions: descriptive names reflecting purpose (e.g., `_verdict_contract_rule()`, `_numbered_list()`)

**Variables:**
- snake_case for variables: `job_id`, `prediction_results`, `step_verdicts`
- Private variables: underscore prefix in class contexts (e.g., `self._metrics_available`, `_cached_client`)
- Constants: UPPER_CASE (e.g., `DEFAULT_TEAM_MEMBERS`, `PRINCIPAL_INVESTIGATOR`)
- Dict keys for JSON/JSON-like structures: snake_case or descriptive (e.g., `"step_verdicts"`, `"key_findings"`, `"clash_score"`)

**Classes:**
- PascalCase: `WorkflowContext`, `AgentResult`, `BaseAgent`, `LLMAgent`, `CompositeEvaluator`
- Abstract base classes: `Base` prefix (e.g., `BaseAgent`, `BaseDesigner`, `BaseMetric`)
- Exception classes: end with `Error` (e.g., `InstallationError`, `PredictorNotFoundError`, `EvaluationError`)
- Config dataclasses: end with `Config` (e.g., `ColabFoldConfig`, `Chai1Config`, `LLMConfig`)

**Type variables:**
- Use `|` for union types in modern Python 3.10+ code (e.g., `LLMAgent | None`)
- Use `Optional[]` for nullable types in some older patterns
- Use `Literal[]` for constrained string values (e.g., `Literal["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"]`)

## Code Style

**Formatting:**
- Tool: Black (via pre-commit hook)
- Line length: 100 characters (configured in `pyproject.toml`)
- Target versions: Python 3.10, 3.11

**Linting:**
- Tool: Ruff with `--fix` flag
- Runs via pre-commit hook
- Config: `pyproject.toml` [tool.ruff] section
- line_length = 100
- Conventions enforced: trailing whitespace, end-of-file fixer, YAML validation, conflict markers

**Type checking:**
- Tool: MyPy
- Config: `pyproject.toml` [tool.mypy]
- Enabled: `warn_return_any = true`, `warn_unused_configs = true`
- Ignores missing imports to handle optional dependencies

## Import Organization

**Order:**
1. `from __future__ import annotations` (at top if using forward references)
2. Standard library imports (e.g., `import json`, `from pathlib import Path`, `from typing import ...`)
3. Third-party imports (e.g., `from pydantic import ...`, `from openai import OpenAI`)
4. Local application imports (e.g., `from protein_design_hub.agents.base import ...`)
5. Conditional/lazy imports within functions (for circular dependency avoidance)

**Examples:**
```python
# from llm_guided.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.llm_agent import LLMAgent
from protein_design_hub.agents.meeting import run_meeting
```

**Path Aliases:**
- No explicit path aliases configured; uses standard `protein_design_hub` package namespace
- All imports use full relative path from package root: `from protein_design_hub.agents.llm_agent import ...`
- Circular imports avoided via lazy imports inside functions (see `orchestrator.py` lines 64-77)

## Error Handling

**Patterns:**
- Custom exceptions defined in `src/protein_design_hub/core/exceptions.py`
- Raise specific exceptions with descriptive messages: `raise InstallationError(self.name, "message", original_error=e)`
- Try-except in strategy code (e.g., auto-install attempts) with fallback to raise
- CLI commands use `raise typer.Exit(0)` for success, `raise typer.Exit(1)` for failure
- Agent failures return `AgentResult.fail(message, error=exception)` instead of raising

**Examples:**
```python
# from base.py: design tools
try:
    self.installer.ensure_installed(auto_update=False)
except Exception as e:
    raise InstallationError(self.name, "Auto-install failed", original_error=e)

# from orchestrator.py: agent pipeline
if not result.success and self.stop_on_failure:
    return result  # Pass failure through context, don't raise
```

## Logging

**Framework:** Python standard `logging` module (not printed to console by default)

**Patterns:**
- Optional logging import in meeting/LLM code:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info(f"Agent {agent_name} took {elapsed:.2f}s")
  ```
- Progress callbacks instead of logging for user-facing updates: `self.progress_callback(stage, item, current, total)`
- Rich library used for CLI output formatting (imported in CLI commands)
- No global logging configuration in library code; upstream caller (CLI/web) configures

## Comments

**When to Comment:**
- Module docstrings: Always (at top of file after imports)
- Class docstrings: For public classes (BaseAgent, WorkflowContext, etc.)
- Function docstrings: For public functions with parameters/return values
- Inline comments: Only when logic is non-obvious or historical
- Internal docstrings: For helper functions that explain algorithm/domain concepts

**JSDoc/TSDoc:**
- Not used (Python project, no TypeScript)
- Docstrings follow Google style within triple quotes:
  ```python
  def run_meeting(
      meeting_type: str,
      agenda: str,
      save_dir: Path,
      ...
  ) -> str | None:
      """Run a meeting and return its summary.

      Args:
          meeting_type: Type of meeting (e.g., "team", "individual")
          agenda: Meeting agenda description
          save_dir: Directory to save discussion transcript

      Returns:
          Meeting summary text or None if no return requested
      """
  ```

## Comments Example from Codebase

See `src/protein_design_hub/agents/meeting.py` lines 14-21 for Performance notes documentation style:
```python
"""Meeting runner for LLM agent discussions.

Supports two meeting types modelled after the Virtual-Lab pattern:
  * **team**:       team lead + N members discuss an agenda in rounds.
  * **individual**: one agent + scientific critic iterate.

Performance notes
-----------------
* The OpenAI client is cached as a module-level singleton so TCP
  connections are reused across the ~70 LLM calls in a full pipeline.
* Ollama GPU checks use a 60 s TTL cache (see ``ollama_gpu.py``).
* Per-agent call timing is printed so bottlenecks are visible.
"""
```

## Function Design

**Size:**
- Functions typically 10-50 lines
- Complex pipelines broken into helpers
- Example: `run_meeting()` in `meeting.py` (~80 lines) handles orchestration, delegates to helpers

**Parameters:**
- Use type hints on all parameters and return values
- Defaults provided where sensible (e.g., `temperature: float | None = None`)
- Optional parameters typed with `Optional[]` or `| None`
- Multiple related parameters grouped in dataclasses (e.g., `WorkflowContext`, `DesignInput`)
- **kwargs used sparingly (e.g., in agent base classes for extensibility)

**Return Values:**
- Always typed (e.g., `-> AgentResult`, `-> str | None`, `-> Dict[str, Any]`)
- Dataclass returns for complex values (e.g., `DesignResult`, `EvaluationResult`, `AgentResult`)
- Use `@classmethod` factory methods for construction: `AgentResult.ok(context, message)`, `AgentResult.fail(message, error)`
- Single return type, not union of value + error (use exception or result object pattern)

## Module Design

**Exports:**
- No `__all__` declarations currently; all public classes/functions importable
- Private modules (leading `_`) not used; private functions used instead (leading underscore)
- Imports at top of files, not scattered through code

**Barrel Files:**
- `__init__.py` files minimal: mostly empty or simple re-exports
- Example: `src/protein_design_hub/agents/__init__.py` may re-export key classes
- No circular dependencies; lazy imports used where needed

**Example Module Structure** (`src/protein_design_hub/agents/llm_guided.py`):
```python
# Module docstring with overview
# Imports (stdlib, 3rd-party, local)
# Mixin class (shared code)
# Helper functions (private, prefixed with _)
# Public agent classes
```

## Dataclass Patterns

**Dataclass Usage:**
- Frozen dataclasses for immutable value objects: `@dataclass(frozen=True)` (e.g., `LLMAgent`)
- Mutable dataclasses for workflow state: `@dataclass` (e.g., `WorkflowContext`, `AgentResult`)
- Default factories for mutable defaults: `field(default_factory=list)`, `field(default_factory=dict)`
- Field descriptions: `Field(default=..., description="...")`  for Pydantic BaseModel classes
- Post-init validation: `__post_init__()` (e.g., in `Sequence` class to validate amino acids)

**Example:**
```python
@dataclass
class WorkflowContext:
    """Shared context for the prediction pipeline."""

    # Identifiers and paths
    job_id: str = ""
    input_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))

    # Step results
    sequences: List[Sequence] = field(default_factory=list)
    prediction_results: Dict[str, PredictionResult] = field(default_factory=dict)

    def with_job_dir(self) -> Path:
        """Return job directory, creating it if needed."""
        ...
```

## Pydantic/BaseSettings Patterns

**Config classes:**
- Inherit from `pydantic.BaseModel` for validation models (e.g., `ColabFoldConfig`)
- Inherit from `pydantic_settings.BaseSettings` for app settings with env var support (e.g., `Settings`)
- Use `Field(default=..., ge=1, le=5, description="...")` for constrained numeric values
- Use `Literal[...]` for enum-like string choices
- Config via `SettingsConfigDict` with `env_file` and other settings

**Example:**
```python
class ColabFoldConfig(BaseModel):
    enabled: bool = True
    num_models: int = Field(default=5, ge=1, le=5, description="Number of models (1-5)")
    msa_mode: Literal["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"] = Field(
        default="mmseqs2_uniref_env"
    )
```

## Testing Conventions

**Test naming:** `test_` prefix (e.g., `test_step_pipeline_has_5_computational_steps()`)
**Test discovery:** Pytest collects `tests/test_*.py` files by default
**Parametrized tests:** `@pytest.mark.parametrize("param_name", values, ids=...)` (e.g., in test_web_smoke.py)
**Fixtures:** via `monkeypatch` (built-in Pytest fixture), `tmp_path` (temp directory fixture)

---

*Convention analysis: 2026-02-21*
