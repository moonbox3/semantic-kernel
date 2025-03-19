# Copyright (c) Microsoft. All rights reserved.

from abc import ABC

from autogen_core import RoutedAgent
from pydantic import model_validator

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class AgentContainerBaseMetaClass(type(KernelBaseModel), type(RoutedAgent)):
    """Meta class for Semantic Kernel agents."""


class AgentContainerBase(KernelBaseModel, RoutedAgent, ABC, metaclass=AgentContainerBaseMetaClass):
    """A base agent container for multi-agent pattern running on Agent runtime."""

    agent: Agent | None = None

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

    @model_validator(mode="after")
    def validate_agent(self) -> "AgentContainerBase":
        """Validate the agent.

        For now we only supports ChatCompletionAgent since our agents have different invocation patterns.
        """
        if not self.agent or isinstance(self.agent, ChatCompletionAgent):
            return self
        raise ValueError("Agent must be a ChatCompletionAgent.")
