# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from typing import ClassVar

from autogen_core import MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.patterns.agent_container import AgentContainerBase
from semantic_kernel.agents.patterns.pattern_base import MultiAgentPatternBase
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


class ConcurrentAgentContainer(AgentContainerBase):
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

        await self.publish_message(ConcurrentResponseMessage(body=response.message), ctx.topic_id)


class CollectionAgentContainer(AgentContainerBase):
    """A agent container for collection results from concurrent agents."""

    def __init__(self, **kwargs):
        """Initialize the collection agent container."""
        super().__init__(description="A container to collect responses from concurrent agents.", **kwargs)

    @message_handler
    async def _handle_message(self, message: ConcurrentResponseMessage, ctx: MessageContext) -> None:
        print(f"From {ctx.sender}: {message.body.content}")


class ConcurrentPattern(MultiAgentPatternBase):
    """A concurrent multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)

    COLLECTION_AGENT_TYPE: ClassVar[str] = "concurrent_collection_container"

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the concurrent pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)

        should_stop = True
        try:
            runtime.start()
        except Exception:
            should_stop = False
            logger.warning("Runtime is already started outside of the pattern.")

        await runtime.publish_message(
            ConcurrentRequestMessage(body=message),
            topic_id=TopicId(self.shared_topic_type, "default"),
        )

        if should_stop:
            await runtime.stop_when_idle()

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            ConcurrentAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent: ConcurrentAgentContainer(agent, shared_topic_type=self.shared_topic_type),
            )
            for agent in self.agents
        ])
        await CollectionAgentContainer.register(
            runtime,
            self.COLLECTION_AGENT_TYPE,
            lambda: CollectionAgentContainer(shared_topic_type=self.shared_topic_type),
        )

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        await runtime.add_subscription(
            TypeSubscription(
                self.shared_topic_type,
                self.COLLECTION_AGENT_TYPE,
            )
        )
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    self.shared_topic_type,
                    self._get_container_type(agent),
                )
            )
            for agent in self.agents
        ])

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_concurrent_container"
