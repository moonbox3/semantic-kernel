# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from typing import ClassVar

from autogen_core import MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.patterns.core.agent_container import AgentContainerBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)

# This topic is the shared topic for all agents in the group chat.
GROUP_CHAT_TOPIC = "GroupChatTopic"


class GroupChatRequestMessage(KernelBaseModel):
    """A request message type for agents in a group chat."""

    pass


class GroupChatResponseMessage(KernelBaseModel):
    """A response message type from agents in a group chat."""

    body: ChatMessageContent


class GroupChatResetMessage(KernelBaseModel):
    """A message to reset a participant's chat history in a group chat."""

    pass


class GroupChatAgentContainer(AgentContainerBase):
    """A agent container for agents that process messages in a group chat."""

    chat_history: ChatHistory = Field(default_factory=ChatHistory)

    @message_handler
    async def _on_group_chat_reset(self, message: GroupChatResetMessage, ctx: MessageContext) -> None:
        self.chat_history.clear()

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=f"Transferred to {message.body.name}",
            )
        )
        self.chat_history.add_message(message.body)

    @message_handler
    async def _on_request_to_speak(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(
            f"Group chat container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        # Add a system message to steer the agent to respond more closely to the instructions.
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=f"Transferred to {self.name}, adopt the persona immediately.",
            )
        )
        response = await self.get_response(self.chat_history)

        logger.debug(
            f"Group chat container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.publish_message(GroupChatResponseMessage(body=response), TopicId(GROUP_CHAT_TOPIC, self.id.key))


class GroupChatManager(KernelBaseModel, ABC):
    """A group chat manager that manages the participants in a group chat."""

    @abstractmethod
    async def should_terminate(self) -> bool:
        """Check if the group chat should terminate."""
        raise NotImplementedError

    @abstractmethod
    async def select_next_agent(self, participant_descriptions: dict[str, str]) -> str:
        """Select the next agent to speak."""
        raise NotImplementedError


class RoundRobinGroupChatManager(GroupChatManager):
    """A round-robin group chat manager."""

    current_index: int = 0
    current_round: int = 0
    max_rounds: int | None = None

    @override
    async def should_terminate(self) -> bool:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None:
            return self.current_round > self.max_rounds
        return False

    @override
    async def select_next_agent(self, participant_descriptions: dict[str, str]) -> str:
        """Select the next agent to speak."""
        next_agent = list(participant_descriptions.keys())[self.current_index]
        self.current_index = (self.current_index + 1) % len(participant_descriptions)
        self.current_round += 1
        return next_agent


class GroupChatManagerContainer(GroupChatAgentContainer):
    """A group chat manager container."""

    manager: GroupChatManager

    participant_descriptions: dict[str, str]
    participant_topics: dict[str, str]

    def __init__(
        self,
        manager: GroupChatManager,
        participant_descriptions: dict[str, str],
        participant_topics: dict[str, str],
        **kwargs,
    ):
        """Initialize the group chat manager container."""
        super().__init__(
            manager=manager,
            participant_descriptions=participant_descriptions,
            participant_topics=participant_topics,
            **kwargs,
        )

    @override
    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        await super()._on_group_chat_message(message, ctx)

        should_terminate = await self.manager.should_terminate()
        if should_terminate:
            logger.debug("Group chat manager decided to terminate the group chat.")
            return

        next_agent = self.manager.select_next_agent(self.participant_descriptions)
        logger.debug(f"Group chat manager selected next agent: {next_agent}")

        await self.publish_message(
            GroupChatRequestMessage(),
            TopicId(self.manager.participant_topics[next_agent], self.id.key),
        )

    @override
    @message_handler
    async def _on_request_to_speak(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        raise RuntimeError("Group chat manager should not receive request to speak messages.")


class GroupChatPattern(KernelBaseModel):
    """A group chat multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)
    runtime: SingleThreadedAgentRuntime

    MANAGER_TYPE: ClassVar[str] = "group_chat_manager_container"

    @classmethod
    async def create(
        cls,
        manager: GroupChatManager,
        agents: list[Agent],
        runtime: SingleThreadedAgentRuntime | None = None,
    ) -> "GroupChatPattern":
        """Create a group chat pattern."""
        if runtime is None:
            runtime = SingleThreadedAgentRuntime()

        # Register all agents
        await asyncio.gather(*[
            GroupChatAgentContainer.register(
                runtime,
                cls.get_container_type(agent),
                lambda: GroupChatAgentContainer(agent),
            )
            for agent in agents
        ])
        await GroupChatManagerContainer.register(
            runtime,
            cls.MANAGER_TYPE,
            lambda: GroupChatManagerContainer(
                manager=manager,
                participant_descriptions={agent.name: agent.description for agent in agents},
                participant_topics={agent.name: cls.get_container_topic(agent) for agent in agents},
            ),
        )
        # Add subscriptions
        subscriptions: list[TypeSubscription] = []
        for agent in agents:
            subscriptions.append(TypeSubscription(GROUP_CHAT_TOPIC, cls.get_container_type(agent)))
            subscriptions.append(TypeSubscription(cls.get_container_topic(agent), cls.get_container_type(agent)))
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])
        await runtime.add_subscription(TypeSubscription(GROUP_CHAT_TOPIC, cls.MANAGER_TYPE))

        return cls(agents=agents, runtime=runtime)

    async def start(self, task: str) -> None:
        """Start the group chat pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)

        self.runtime.start()
        await self.runtime.publish_message(
            GroupChatResponseMessage(body=message),
            TopicId(GROUP_CHAT_TOPIC, "default"),
        )
        await self.runtime.stop_when_idle()

    @staticmethod
    def get_container_type(agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_group_chat_container"

    @staticmethod
    def get_container_topic(agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}_group_chat_topic"
