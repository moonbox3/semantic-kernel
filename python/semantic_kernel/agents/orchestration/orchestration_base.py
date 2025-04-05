# Copyright (c) Microsoft. All rights reserved.

import uuid
from abc import ABC, abstractmethod

from autogen_core import SingleThreadedAgentRuntime
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class OrchestrationStartMessage(KernelBaseModel):
    """A orchestration start message type that kicks off the multi-agent orchestration."""

    pass


class OrchestrationResultMessage(KernelBaseModel):
    """A orchestration result message type that contains the result of the multi-agent orchestration."""

    pass


class OrchestrationBase(KernelBaseModel, ABC):
    """Base class for multi-agent orchestration."""

    agents: list[Agent] = Field(default_factory=list)

    shared_topic_type: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The shared topic of the pattern. It needs to be unique for each pattern instance.",
    )

    async def start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the multi-agent orchestration."""
        await self._register_agents(runtime)
        await self._add_subscriptions(runtime)
        await self._start(task, runtime)

    @abstractmethod
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the multi-agent orchestration."""
        pass

    @abstractmethod
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        pass

    @abstractmethod
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        pass
