"""Base agent interface for pipeline steps."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from protein_design_hub.agents.context import WorkflowContext


@dataclass
class AgentResult:
    """Result from a single agent run."""

    success: bool = True
    message: str = ""
    context: Optional[WorkflowContext] = None
    error: Optional[Exception] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def ok(cls, context: WorkflowContext, message: str = "") -> "AgentResult":
        return cls(success=True, message=message, context=context)

    @classmethod
    def fail(cls, message: str, error: Optional[Exception] = None) -> "AgentResult":
        return cls(success=False, message=message, error=error)


class BaseAgent(ABC):
    """
    Base class for pipeline step agents.

    Each agent is responsible for one logical step: it receives a
    WorkflowContext, performs its step, updates the context, and
    returns an AgentResult.
    """

    name: str = "base"
    description: str = "Base agent"

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            progress_callback: Optional callback(stage, item, current, total).
            **kwargs: Agent-specific options.
        """
        self.progress_callback = progress_callback
        self.options = kwargs

    @abstractmethod
    def run(self, context: WorkflowContext) -> AgentResult:
        """
        Execute this agent's step.

        Args:
            context: Current workflow context (read and write).

        Returns:
            AgentResult with updated context on success, or failure info.
        """
        ...

    def _report_progress(self, stage: str, item: str, current: int, total: int) -> None:
        if self.progress_callback:
            self.progress_callback(stage, item, current, total)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
