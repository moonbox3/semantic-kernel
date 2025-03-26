# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from abc import ABC, abstractmethod
from typing import ClassVar

from autogen_core import MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.patterns.agent_container import AgentContainerBase
from semantic_kernel.agents.patterns.pattern_base import MultiAgentPatternBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


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

    chat_history: ChatHistory = Field(
        default_factory=ChatHistory, description="Temporary message storage between invocations."
    )
    agent_thread: AgentThread | None = None

    @message_handler
    async def _on_group_chat_reset(self, message: GroupChatResetMessage, ctx: MessageContext) -> None:
        self.chat_history.clear()
        if self.agent_thread:
            await self.agent_thread.delete()
            self.agent_thread = None
        else:
            logger.warning("Non-existent agent thread cannot be deleted.")

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Transferred to {message.body.name}",
            )
        )
        self.chat_history.add_message(message.body)

    @message_handler
    async def _on_request_to_speak(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        if not self.agent:
            raise RuntimeError("Agent not set for the container. Please provide an agent or override this method.")

        logger.debug(
            f"Group chat container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        # Add a user message to steer the agent to respond more closely to the instructions.
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Transferred to {self.agent.name}, adopt the persona immediately.",
            )
        )
        response = await self.agent.get_response(messages=self.chat_history.messages, thread=self.agent_thread)

        self.chat_history.clear()
        self.agent_thread = response.thread

        logger.debug(
            f"Group chat container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.publish_message(
            GroupChatResponseMessage(body=response.message),
            TopicId(self.shared_topic_type, self.id.key),
        )


class GroupChatManager(KernelBaseModel, ABC):
    """A group chat manager that manages the flow of a group chat."""

    current_round: int = 0
    max_rounds: int | None = None

    @abstractmethod
    async def should_terminate(self, chat_history: ChatHistory) -> bool:
        """Check if the group chat should terminate.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> str:
        """Select the next agent to speak.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        raise NotImplementedError


class RoundRobinGroupChatManager(GroupChatManager):
    """A round-robin group chat manager."""

    current_index: int = 0

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> bool:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None:
            return self.current_round > self.max_rounds
        return False

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> str:
        """Select the next agent to speak."""
        next_agent = list(participant_descriptions.keys())[self.current_index]
        self.current_index = (self.current_index + 1) % len(participant_descriptions)
        self.current_round += 1
        return next_agent


class KernelFunctionGroupChatManager(GroupChatManager):
    """A simple model-based group chat manager."""

    kernel: Kernel

    termination_prompt: str = Field(
        default="""
You are in a role play game. Read the following conversation. Then determine if the game should continue or terminate.

###
                                    
{{$history}}                        

###

Read the above conversation. Then determine if the game should continue or terminate.
Only return "CONTINUE" or "TERMINATE".
"""
    )
    selection_prompt: str = Field(
        default="""
You are in a role play game. The following roles are available:
{{$roles}}.
Read the following conversation. Then select the next role from {{$participants}} to play. Only return the role.

###

{{$history}}

###

Read the above conversation. Then select the next role from {{$participants}} to play. Only return the name of the role.
"""
    )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> bool:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None and self.current_round >= self.max_rounds:
            logger.debug("Group chat manager reached the maximum number of rounds.")
            return True

        self.current_round += 1

        response = await self.kernel.invoke_prompt(
            self.termination_prompt,
            arguments=KernelArguments(history=chat_history.to_prompt()),
        )
        if response is None:
            raise RuntimeError("No response received from the kernel.")
        if (
            not isinstance(response.value, list)
            or len(response.value) < 1
            or not isinstance(response.value[0], ChatMessageContent)
        ):
            raise RuntimeError("Invalid response received from the kernel.")

        return "terminate" in response.value[0].content.lower()

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> str:
        """Select the next agent to speak."""
        response = await self.kernel.invoke_prompt(
            self.selection_prompt,
            arguments=KernelArguments(
                history=chat_history.to_prompt(),
                roles=", ".join(participant_descriptions.values()),
                participants=", ".join(participant_descriptions.keys()),
            ),
        )
        if response is None:
            raise RuntimeError("No response received from the kernel.")
        if (
            not isinstance(response.value, list)
            or len(response.value) < 1
            or not isinstance(response.value[0], ChatMessageContent)
        ):
            raise RuntimeError("Invalid response received from the kernel.")

        for participant_name in participant_descriptions:
            if participant_name.lower() in response.value[0].content.lower():
                return participant_name

        raise RuntimeError(f"Unknown participant selected: {response.value[0].content}.")


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
            description="A container for the group chat manager.",
            **kwargs,
        )

    @override
    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        await super()._on_group_chat_message(message, ctx)

        should_terminate = await self.manager.should_terminate(self.chat_history)
        if should_terminate:
            logger.debug("Group chat manager decided to terminate the group chat.")
            return

        next_agent = await self.manager.select_next_agent(self.chat_history, self.participant_descriptions)
        logger.debug(f"Group chat manager selected agent: {next_agent} on round {self.manager.current_round}.")

        await self.publish_message(
            GroupChatRequestMessage(),
            TopicId(self.participant_topics[next_agent], self.id.key),
        )

    @override
    @message_handler
    async def _on_request_to_speak(self, message: GroupChatRequestMessage, ctx: MessageContext) -> None:
        raise RuntimeError("Group chat manager should not receive request to speak messages.")


class GroupChatPattern(MultiAgentPatternBase):
    """A group chat multi-agent pattern."""

    agents: list[Agent] = Field(default_factory=list)
    manager: GroupChatManager

    MANAGER_TYPE: ClassVar[str] = "group_chat_manager_container"

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the group chat pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task, name="User")

        should_stop = True
        try:
            runtime.start()
        except Exception:
            should_stop = False
            logger.warning("Runtime is already started outside of the pattern.")

        await runtime.publish_message(
            GroupChatResponseMessage(body=message),
            TopicId(self.shared_topic_type, "default"),
        )

        if should_stop:
            await runtime.stop_when_idle()

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            GroupChatAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent: GroupChatAgentContainer(
                    agent,
                    description=agent.description,
                    shared_topic_type=self.shared_topic_type,
                ),
            )
            for agent in self.agents
        ])
        await GroupChatManagerContainer.register(
            runtime,
            self.MANAGER_TYPE,
            lambda: GroupChatManagerContainer(
                manager=self.manager,
                participant_descriptions={agent.name: agent.description for agent in self.agents},
                participant_topics={agent.name: self._get_container_topic(agent) for agent in self.agents},
                shared_topic_type=self.shared_topic_type,
            ),
        )

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        subscriptions: list[TypeSubscription] = []
        for agent in self.agents:
            subscriptions.append(TypeSubscription(self.shared_topic_type, self._get_container_type(agent)))
            subscriptions.append(TypeSubscription(self._get_container_topic(agent), self._get_container_type(agent)))
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])
        await runtime.add_subscription(TypeSubscription(self.shared_topic_type, self.MANAGER_TYPE))

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_group_chat_container"

    def _get_container_topic(self, agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}_group_chat_{self.shared_topic_type}"
