# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from typing import Annotated, ClassVar

from autogen_core import MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.patterns.core.agent_container import AgentContainerBase
from semantic_kernel.agents.patterns.core.pattern_base import MultiAgentPatternBase
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


class SequentialAgentContainer(AgentContainerBase):
    """A agent container for sequential agents that process tasks."""

    sequential_topic_type: Annotated[str | None, "The topic of the next agent in the sequence."]

    def __init__(self, agent: Agent, sequential_topic_type: str | None) -> None:
        """Initialize the agent container."""
        super().__init__(agent=agent, sequential_topic_type=sequential_topic_type)

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

        if self.sequential_topic_type:
            await self.publish_message(
                SequentialRequestMessage(body=response.message),
                TopicId(self.sequential_topic_type, self.id.key),
            )


class CollectionAgentContainer(AgentContainerBase):
    """A agent container for collection results from the last agent in the sequence."""

    def __init__(self):
        """Initialize the collection agent container."""
        super().__init__(description="A container to collect responses from the last agent in the sequence.")

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        print(f"From {ctx.sender}: {message.body.content}")


class SequentialPattern(MultiAgentPatternBase):
    """A sequential multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)

    COLLECTION_AGENT_TYPE: ClassVar[str] = "sequential_collection_container"
    COLLECTION_AGENT_TOPIC: ClassVar[str] = "sequential_collection_container_topic"

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the sequential pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)

        should_stop = True
        try:
            runtime.start()
        except Exception:
            should_stop = False
            logger.warning("Runtime is already started outside of the pattern.")

        await runtime.publish_message(
            SequentialRequestMessage(body=message),
            TopicId(self._get_container_topic(self.agents[0]), "default"),
        )

        if should_stop:
            await runtime.stop_when_idle()

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await CollectionAgentContainer.register(
            runtime,
            self.COLLECTION_AGENT_TYPE,
            lambda: CollectionAgentContainer(),
        )
        await asyncio.gather(*[
            SequentialAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent: SequentialAgentContainer(
                    agent,
                    self._get_container_topic(self.agents[index + 1])
                    if index + 1 < len(self.agents)
                    else self.COLLECTION_AGENT_TOPIC,
                ),
            )
            for index, agent in enumerate(self.agents)
        ])

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        await runtime.add_subscription(
            TypeSubscription(
                self.COLLECTION_AGENT_TOPIC,
                self.COLLECTION_AGENT_TYPE,
            )
        )
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
        return f"{agent.name}_sequential_topic"
