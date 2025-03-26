# Copyright (c) Microsoft. All rights reserved.

from abc import ABC

from autogen_core import RoutedAgent
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class AgentContainerBaseMetaClass(type(KernelBaseModel), type(RoutedAgent)):
    """Meta class for Semantic Kernel agents."""


class AgentContainerBase(KernelBaseModel, RoutedAgent, ABC, metaclass=AgentContainerBaseMetaClass):
    """A base agent container for multi-agent orchestration running on Agent runtime."""

    agent: Agent | None = Field(default=None, description="An agent to be run in the container.")
    shared_topic_type: str = Field(
        description="The unique topic of the orchestration pattern that the container is participating in."
    )

    def __init__(self, agent: Agent | None = None, description: str | None = None, **kwargs):
        """Initialize the agent container.

        Args:
            agent (Agent | None): An agent to be run in the container.
            description (str | None): A description of the agent container.
            **kwargs: Additional keyword arguments.
        """
        KernelBaseModel.__init__(self, agent=agent, **kwargs)

        if not agent and not description:
            raise ValueError("Either agent or description must be provided.")
        RoutedAgent.__init__(self, description=description or agent.description)
