# Copyright (c) Microsoft. All rights reserved.

from abc import ABC, abstractmethod

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.kernel_pydantic import KernelBaseModel


class MultiAgentPatternBase(KernelBaseModel, ABC):
    """Base class for multi-agent patterns."""

    async def start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the multi-agent pattern."""
        await self._register_agents(runtime)
        await self._add_subscriptions(runtime)
        await self._start(task, runtime)

    @abstractmethod
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the multi-agent pattern."""
        pass

    @abstractmethod
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        pass

    @abstractmethod
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        pass
