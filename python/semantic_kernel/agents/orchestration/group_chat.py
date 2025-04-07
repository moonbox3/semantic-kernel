# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from autogen_core import (
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    message_handler,
)
from pydantic import Field

from semantic_kernel.agents.agent import Agent, AgentThread
from semantic_kernel.agents.orchestration.container_base import ContainerBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase, OrchestrationResultMessage
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
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


class GroupChatAgentContainer(ContainerBase):
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
        if message.body.role != AuthorRole.USER:
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
            TopicId(self.internal_topic_type, self.id.key),
        )


class BoolWithReason(KernelBaseModel):
    """A class to represent a boolean value with a reason."""

    value: bool
    reason: str

    def __bool__(self) -> bool:
        """Return the boolean value."""
        return self.value


class StringWithReason(KernelBaseModel):
    """A class to represent a string value with a reason."""

    value: str
    reason: str

    def __str__(self) -> str:
        """Return the string value."""
        return self.value


class ObjectWithReason(KernelBaseModel):
    """A class to represent an object value with a reason."""

    value: object
    reason: str

    def __str__(self) -> str:
        """Return the string value."""
        return str(self.value)


class GroupChatManager(KernelBaseModel, ABC):
    """A group chat manager that manages the flow of a group chat."""

    current_round: int = 0
    max_rounds: int | None = None

    user_input_func: Callable[[ChatHistory], Awaitable[str]] | None = None

    @abstractmethod
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
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
    ) -> StringWithReason:
        """Select the next agent to speak.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        raise NotImplementedError

    @abstractmethod
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ObjectWithReason:
        """Filter the results of the group chat.

        Args:
            chat_history (ChatHistory): The chat history of the group chat.
            participant_descriptions (dict[str, str]): The descriptions of the participants in the group chat.
        """
        raise NotImplementedError


class RoundRobinGroupChatManager(GroupChatManager):
    """A round-robin group chat manager."""

    current_index: int = 0

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input."""
        return BoolWithReason(
            value=False,
            reason="The default round-robin group chat manager does not request user input.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None:
            return BoolWithReason(
                value=self.current_round > self.max_rounds,
                reason="Maximum rounds reached.",
            )
        return BoolWithReason(value=False, reason="No maximum rounds set.")

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringWithReason:
        """Select the next agent to speak."""
        next_agent = list(participant_descriptions.keys())[self.current_index]
        self.current_index = (self.current_index + 1) % len(participant_descriptions)
        self.current_round += 1
        return StringWithReason(value=next_agent, reason="Round-robin selection.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ObjectWithReason:
        """Filter the results of the group chat."""
        return ObjectWithReason(
            value=chat_history.messages[-1],
            reason="The last message in the chat history is the result in the default round-robin group chat manager.",
        )


class KernelFunctionGroupChatManager(GroupChatManager):
    """A simple model-based group chat manager.

    This manager requires a model that supports structured output.
    """

    kernel: Kernel

    request_user_input_prompt: str = Field(
        default="""
You are in a role play game. Read the following conversation. Then determine if the game should request user input.
###

{{$history}}

###

Read the above conversation. Then determine if the game should request user input.
"""
    )

    termination_prompt: str = Field(
        default="""
You are in a role play game. Read the following conversation. Then determine if the game should terminate or not.

###
                                    
{{$history}}                        

###

Read the above conversation. Then determine if the game should terminate or not.
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
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should request user input."""
        if self.user_input_func is None:
            return False

        response = await self.kernel.invoke_prompt(
            self.request_user_input_prompt,
            arguments=KernelArguments(
                settings=PromptExecutionSettings(response_format=BoolWithReason),
                history=chat_history.to_prompt(),
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

        return BoolWithReason.model_validate_json(response.value[0].content)

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        """Check if the group chat should terminate."""
        if self.max_rounds is not None and self.current_round >= self.max_rounds:
            logger.debug("Group chat manager reached the maximum number of rounds.")
            return True

        self.current_round += 1

        response = await self.kernel.invoke_prompt(
            self.termination_prompt,
            arguments=KernelArguments(
                settings=PromptExecutionSettings(response_format=BoolWithReason),
                history=chat_history.to_prompt(),
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

        return BoolWithReason.model_validate_json(response.value[0].content)

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringWithReason:
        """Select the next agent to speak."""
        response = await self.kernel.invoke_prompt(
            self.selection_prompt,
            arguments=KernelArguments(
                settings=PromptExecutionSettings(response_format=StringWithReason),
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

        participant_name_with_reason = StringWithReason.model_validate_json(response.value[0].content)

        for participant_name in participant_descriptions:
            if participant_name.lower() in participant_name_with_reason.value.lower():
                return participant_name_with_reason

        raise RuntimeError(f"Unknown participant selected: {response.value[0].content}.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ObjectWithReason:
        """Filter the results of the group chat."""
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        return ObjectWithReason(
            value=chat_history.messages[-1],
            reason="The last message in the chat history is the result by default.",
        )


class GroupChatManagerContainer(RoutedAgent):
    """A group chat manager container."""

    def __init__(
        self,
        manager: GroupChatManager,
        internal_topic_type: str,
        participant_descriptions: dict[str, str],
        participant_topics: dict[str, str],
    ):
        """Initialize the group chat manager container."""
        self._manager = manager
        self._internal_topic_type = internal_topic_type
        self._chat_history = ChatHistory()
        self._participant_descriptions = participant_descriptions
        self._participant_topics = participant_topics

        super().__init__(description="A container for the group chat manager.")

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
        if message.body.role != AuthorRole.USER:
            self._chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=f"Transferred to {message.body.name}",
                )
            )
        self._chat_history.add_message(message.body)

        should_request_user_input = await self._manager.should_request_user_input(self._chat_history)
        if should_request_user_input and self._manager.user_input_func:
            logger.debug(f"Group chat manager requested user input. Reason: {should_request_user_input.reason}")
            user_input = await self._manager.user_input_func(self._chat_history)
            if user_input:
                self._chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=user_input))
                await self.publish_message(
                    GroupChatResponseMessage(body=ChatMessageContent(role=AuthorRole.USER, content=user_input)),
                    TopicId(self._internal_topic_type, self.id.key),
                )
                logger.debug("User input received and added to chat history.")

        should_terminate = await self._manager.should_terminate(self._chat_history)
        if should_terminate:
            logger.debug(f"Group chat manager decided to terminate the group chat. Reason: {should_terminate.reason}")
            result = await self._manager.filter_results(self._chat_history)
            await self.publish_message(
                OrchestrationResultMessage(body=result),
                TopicId(self._internal_topic_type, self.id.key),
            )
            return

        next_agent = await self._manager.select_next_agent(self._chat_history, self._participant_descriptions)
        logger.debug(
            f"Group chat manager selected agent: {next_agent} on round {self._manager.current_round}. "
            f"Reason: {next_agent.reason}"
        )

        await self.publish_message(
            GroupChatRequestMessage(),
            TopicId(self._participant_topics[str(next_agent)], self.id.key),
        )


class GroupChatOrchestration(OrchestrationBase):
    """A group chat multi-agent pattern orchestration."""

    manager: GroupChatManager

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the group chat pattern."""
        group_manager_type = f"GroupManager_{uuid.uuid4().hex}"
        await GroupChatManagerContainer.register(
            runtime,
            group_manager_type,
            lambda: GroupChatManagerContainer(
                manager=self.manager,
                internal_topic_type=self.internal_topic_type,
                participant_descriptions={agent.name: agent.description for agent in self.agents},
                participant_topics={agent.name: self._get_container_topic(agent) for agent in self.agents},
            ),
        )
        await runtime.add_subscription(TypeSubscription(self.internal_topic_type, group_manager_type))

        message = ChatMessageContent(AuthorRole.USER, content=task, name="User")

        await runtime.publish_message(
            GroupChatResponseMessage(body=message),
            TopicId(self.internal_topic_type, "default"),
        )

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
                    internal_topic_type=self.internal_topic_type,
                ),
            )
            for agent in self.agents
        ])

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        subscriptions: list[TypeSubscription] = []
        for agent in self.agents:
            subscriptions.append(TypeSubscription(self.internal_topic_type, self._get_container_type(agent)))
            subscriptions.append(TypeSubscription(self._get_container_topic(agent), self._get_container_type(agent)))
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_group_chat_container"

    def _get_container_topic(self, agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}_group_chat_{self.internal_topic_type}"
