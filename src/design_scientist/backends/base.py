"""Backend interfaces for model calls and workspace agents."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from design_scientist.schemas import (
    ModelRequest,
    ModelResponse,
    WorkspaceAgentResult,
    WorkspaceAgentTask,
)


@runtime_checkable
class ModelBackend(Protocol):
    """Interface for text or structured model generation."""

    @abstractmethod
    def complete(self, request: ModelRequest) -> ModelResponse:
        """Run a model request and return the model response."""


@runtime_checkable
class WorkspaceAgentBackend(Protocol):
    """Interface for controlled workspace-level agent tasks."""

    @abstractmethod
    def run_task(self, task: WorkspaceAgentTask) -> WorkspaceAgentResult:
        """Run a workspace task and return the agent result."""
