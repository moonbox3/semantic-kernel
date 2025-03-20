# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import Annotated, ClassVar

from autogen_core import MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.patterns.core.agent_container import AgentContainerBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

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

        chat_history = ChatHistory(messages=[message.body])
        response = await self.agent.get_response(chat_history)

        logger.debug(
            f"Sequential container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        if self.sequential_topic_type:
            await self.publish_message(
                SequentialRequestMessage(body=response),
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


class SequentialPattern(KernelBaseModel):
    """A sequential multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)
    runtime: SingleThreadedAgentRuntime

    COLLECTION_AGENT_TYPE: ClassVar[str] = "sequential_collection_container"
    COLLECTION_AGENT_TOPIC: ClassVar[str] = "sequential_collection_container_topic"

    @classmethod
    async def create(
        cls, agents: list[Agent], runtime: SingleThreadedAgentRuntime | None = None
    ) -> "SequentialPattern":
        """Create a sequential pattern."""
        if runtime is None:
            runtime = SingleThreadedAgentRuntime()

        # Register all agents
        await CollectionAgentContainer.register(
            runtime,
            cls.COLLECTION_AGENT_TYPE,
            lambda: CollectionAgentContainer(),
        )
        await asyncio.gather(*[
            SequentialAgentContainer.register(
                runtime,
                cls.get_container_type(agent),
                lambda: SequentialAgentContainer(
                    agent,
                    cls.get_container_topic(agents[index + 1])
                    if index + 1 < len(agents)
                    else cls.COLLECTION_AGENT_TOPIC,
                ),
            )
            for index, agent in enumerate(agents)
        ])
        # Add subscriptions
        await runtime.add_subscription(
            TypeSubscription(
                cls.COLLECTION_AGENT_TOPIC,
                cls.COLLECTION_AGENT_TYPE,
            )
        )
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    cls.get_container_topic(agent),
                    cls.get_container_type(agent),
                )
            )
            for agent in agents
        ])

        return cls(agents=agents, runtime=runtime)

    async def start(self, task: str) -> None:
        """Start the sequential pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)

        self.runtime.start()
        await self.runtime.publish_message(
            SequentialRequestMessage(body=message),
            TopicId(SequentialPattern.get_container_topic(self.agents[0]), "default"),
        )
        await self.runtime.stop_when_idle()

    @staticmethod
    def get_container_type(agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_sequential_container"

    @staticmethod
    def get_container_topic(agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}_sequential_topic"
