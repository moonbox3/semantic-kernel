# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
import uuid

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.container_base import ContainerBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase, OrchestrationResultMessage
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


class SequentialRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class SequentialAgentContainer(ContainerBase):
    """A agent container for sequential agents that process tasks."""

    def __init__(self, agent: Agent, **kwargs) -> None:
        """Initialize the agent container."""
        super().__init__(agent=agent, **kwargs)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(
            f"Sequential container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        response = await self.agent.get_response(messages=message.body)

        logger.debug(
            f"Sequential container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.publish_message(
            SequentialRequestMessage(body=response.message),
            TopicId(self.internal_topic_type, self.id.key),
        )


class CollectionAgent(RoutedAgent):
    """A agent container for collection results from the last agent in the sequence."""

    def __init__(self, description: str, internal_topic_type: str) -> None:
        """Initialize the collection agent container."""
        self._internal_topic_type = internal_topic_type
        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        await self.publish_message(
            OrchestrationResultMessage(body=message.body),
            TopicId(self._internal_topic_type, self.id.key),
        )


class SequentialOrchestration(OrchestrationBase):
    """A sequential multi-agent pattern orchestration."""

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the sequential pattern."""
        collection_agent_type = f"Collection_{uuid.uuid4().hex}"
        await CollectionAgent.register(
            runtime,
            collection_agent_type,
            lambda: CollectionAgent(
                description="An internal agent that is responsible for collection results",
                internal_topic_type=self.internal_topic_type,
            ),
        )
        await runtime.add_subscription(TypeSubscription(self._get_collection_agent_topic(), collection_agent_type))

        message = ChatMessageContent(AuthorRole.USER, content=task)
        await runtime.publish_message(
            SequentialRequestMessage(body=message),
            TopicId(self._get_container_topic(self.agents[0]), "default"),
        )

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            SequentialAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent, index=index: SequentialAgentContainer(
                    agent,
                    internal_topic_type=self._get_container_topic(self.agents[index + 1])
                    if index + 1 < len(self.agents)
                    else self._get_collection_agent_topic(),
                ),
            )
            for index, agent in enumerate(self.agents)
        ])

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    self._get_container_topic(agent),
                    self._get_container_type(agent),
                )
            )
            for agent in self.agents
        ])

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_sequential_container"

    def _get_container_topic(self, agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}_sequential_topic_{self.internal_topic_type}"

    def _get_collection_agent_topic(self) -> str:
        """Get the collection agent topic."""
        return f"Collection_{self.internal_topic_type}"
