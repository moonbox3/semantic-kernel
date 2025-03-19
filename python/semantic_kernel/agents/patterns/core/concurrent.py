# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    SingleThreadedAgentRuntime,
    default_subscription,
    message_handler,
)
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.patterns.core.agent_container import AgentContainerBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

logger: logging.Logger = logging.getLogger(__name__)


class ConcurrentRequestType(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentResponseType(KernelBaseModel):
    """A response message type for concurrent agents."""

    body: ChatMessageContent


@default_subscription
class ConcurrentAgentContainer(AgentContainerBase):
    """A agent container for concurrent agents that process tasks."""

    @message_handler
    async def _handle_message(self, message: ConcurrentRequestType, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        chat_history = ChatHistory(messages=[message.body])
        response = await self.agent.get_response(chat_history)

        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.publish_message(ConcurrentResponseType(body=response), ctx.topic_id)


@default_subscription
class CollectionAgentContainer(AgentContainerBase):
    """A agent container for collection results from concurrent agents."""

    def __init__(self):
        """Initialize the collection agent container."""
        super().__init__(description="A container to collect responses from concurrent agents.")

    @message_handler
    async def _handle_message(self, message: ConcurrentResponseType, ctx: MessageContext) -> None:
        print(f"From {ctx.sender}: {message.body.content}")


class ConcurrentPattern(KernelBaseModel):
    """A concurrent multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)
    runtime: SingleThreadedAgentRuntime

    @classmethod
    async def create(
        cls, agents: list[Agent], runtime: SingleThreadedAgentRuntime | None = None
    ) -> "ConcurrentPattern":
        """Create a concurrent pattern."""
        if runtime is None:
            runtime = SingleThreadedAgentRuntime()

        await asyncio.gather(*[
            ConcurrentAgentContainer.register(
                runtime,
                f"{agent.name}_container",
                lambda: ConcurrentAgentContainer(agent),
            )
            for agent in agents
        ])
        await CollectionAgentContainer.register(
            runtime,
            "collection_container",
            lambda: CollectionAgentContainer(),
        )

        return cls(agents=agents, runtime=runtime)

    async def start(self, task: str) -> None:
        """Start the concurrent pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)

        self.runtime.start()
        await self.runtime.publish_message(ConcurrentRequestType(body=message), topic_id=DefaultTopicId())
        await self.runtime.stop_when_idle()
