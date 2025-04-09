# Copyright (c) Microsoft. All rights reserved.


from autogen_core import RoutedAgent
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase
from semantic_kernel.kernel_pydantic import KernelBaseModel


class ContainerBaseMetaClass(type(KernelBaseModel), type(RoutedAgent)):
    """Meta class for Semantic Kernel agent runtime container."""


class ContainerBase(KernelBaseModel, RoutedAgent, metaclass=ContainerBaseMetaClass):
    """A base agent container for multi-agent orchestration running on Agent runtime."""

    agent: Agent | OrchestrationBase = Field(description="An agent or orchestration to be run in the container.")

    def __init__(self, agent: Agent | OrchestrationBase, **kwargs) -> None:
        """Initialize the agent container.

        Args:
            agent (Agent | None): An agent to be run in the container.
            **kwargs: Additional keyword arguments.
        """
        KernelBaseModel.__init__(self, agent=agent, **kwargs)
        RoutedAgent.__init__(self, description=agent.description)
