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


class ConcurrentRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentResponseMessage(KernelBaseModel):
    """A response message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentAgentContainer(ContainerBase):
    """A agent container for concurrent agents that process tasks."""

    @message_handler
    async def _handle_message(self, message: ConcurrentRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        response = await self.agent.get_response(messages=message.body)

        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.publish_message(
            ConcurrentResponseMessage(body=response.message),
            TopicId(self.internal_topic_type, self.id.key),
        )


class CollectionAgentContainer(RoutedAgent):
    """A agent container for collection results from concurrent agents."""

    def __init__(self, description: str, internal_topic_type: str, expected_answer_count: int) -> None:
        """Initialize the collection agent container."""
        self._internal_topic_type = internal_topic_type
        self._expected_answer_count = expected_answer_count
        self._results: dict[str, ChatMessageContent] = {}

        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: ConcurrentResponseMessage, ctx: MessageContext) -> None:
        self._results[message.body.name] = message.body

        if len(self._results) == self._expected_answer_count:
            logger.debug(f"Collection container (Container ID: {self.id}) finished processing all responses.")
            await self.publish_message(
                OrchestrationResultMessage(body=self._results),
                TopicId(self._internal_topic_type, self.id.key),
            )


class ConcurrentOrchestration(OrchestrationBase):
    """A concurrent multi-agent pattern orchestration."""

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the concurrent pattern."""
        collection_agent_type = f"self.Collection_{uuid.uuid4().hex}"
        await CollectionAgentContainer.register(
            runtime,
            collection_agent_type,
            lambda: CollectionAgentContainer(
                description="An internal agent that is responsible for collection results",
                internal_topic_type=self.internal_topic_type,
                expected_answer_count=len(self.agents),
            ),
        )
        await runtime.add_subscription(TypeSubscription(self.internal_topic_type, collection_agent_type))

        message = ChatMessageContent(AuthorRole.USER, content=task)
        await runtime.publish_message(
            ConcurrentRequestMessage(body=message),
            topic_id=TopicId(self.internal_topic_type, "default"),
        )

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            ConcurrentAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent: ConcurrentAgentContainer(
                    agent,
                    self.internal_topic_type,
                ),
            )
            for agent in self.agents
        ])

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    self.internal_topic_type,
                    self._get_container_type(agent),
                )
            )
            for agent in self.agents
        ])

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_concurrent_container"
