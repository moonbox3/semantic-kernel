# Copyright (c) Microsoft. All rights reserved.


from autogen_core import RoutedAgent
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class ContainerBaseMetaClass(type(KernelBaseModel), type(RoutedAgent)):
    """Meta class for Semantic Kernel agent runtime container."""


class ContainerBase(KernelBaseModel, RoutedAgent, metaclass=ContainerBaseMetaClass):
    """A base agent container for multi-agent orchestration running on Agent runtime."""

    agent: Agent = Field(description="An agent to be run in the container.")
    internal_topic_type: str = Field(
        description="The unique topic of the orchestration pattern that the container is participating in."
    )

    def __init__(self, agent: Agent, internal_topic_type: str, **kwargs) -> None:
        """Initialize the agent container.

        Args:
            agent (Agent | None): An agent to be run in the container.
            internal_topic_type (str): The unique topic of the orchestration pattern that the container is part of.
            **kwargs: Additional keyword arguments.
        """
        KernelBaseModel.__init__(self, agent=agent, internal_topic_type=internal_topic_type, **kwargs)
        RoutedAgent.__init__(self, description=agent.description)
