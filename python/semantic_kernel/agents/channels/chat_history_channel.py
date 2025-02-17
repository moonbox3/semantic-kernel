# Copyright (c) Microsoft. All rights reserved.

import sys
from collections.abc import AsyncIterable
from copy import deepcopy

from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.exceptions.agent_exceptions import AgentInvokeException

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from semantic_kernel.agents.channels.agent_channel import AgentChannel
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.exceptions import ServiceInvalidTypeError
from semantic_kernel.utils.experimental_decorator import experimental_class

if TYPE_CHECKING:
    from semantic_kernel.agents.agent import Agent
    from semantic_kernel.contents.chat_history import ChatHistory


@experimental_class
@runtime_checkable
class ChatHistoryAgentProtocol(Protocol):
    """Contract for an agent that utilizes a ChatHistoryChannel."""

    @abstractmethod
    async def invoke(self, history: "ChatHistory") -> "ChatMessageContent | None":
        """Invoke the chat history agent protocol."""
        ...

    @abstractmethod
    def invoke_stream(self, history: "ChatHistory") -> AsyncIterable["ChatMessageContent"]:
        """Invoke the chat history agent protocol in streaming mode."""
        ...


@experimental_class
class ChatHistoryChannel(AgentChannel, ChatHistory):
    """An AgentChannel specialization for that acts upon a ChatHistoryHandler."""

    ALLOWED_CONTENT_TYPES: ClassVar[tuple[type, ...]] = (
        ImageContent,
        FunctionCallContent,
        FunctionResultContent,
        StreamingTextContent,
        TextContent,
    )

    @override
    async def invoke(
        self,
        agent: "Agent",
        **kwargs: Any,
    ) -> tuple[bool, ChatMessageContent]:
        """Perform a discrete incremental interaction between a single Agent and AgentChat.

        Args:
            agent: The agent to interact with.
            kwargs: The keyword arguments.

        Returns:
            An async iterable of ChatMessageContent.
        """
        if not isinstance(agent, ChatHistoryAgentProtocol):
            agent_id = getattr(agent, "id", "")
            raise ServiceInvalidTypeError(
                f"Invalid channel binding for agent with id: `{agent_id}` of type: ({type(agent).__name__})."
            )

        response_message = await agent.invoke(self)
        if response_message is None:
            raise AgentInvokeException("The agent did not return a message.")
        if response_message is not None:
            self.messages.append(response_message)

        return self._is_message_visible(response_message), response_message

    @override
    async def invoke_stream(
        self, agent: "Agent", messages: list[ChatMessageContent], **kwargs: Any
    ) -> AsyncIterable[ChatMessageContent]:
        """Perform a discrete incremental stream interaction between a single Agent and AgentChat.

        Args:
            agent: The agent to interact with.
            messages: The history of messages in the conversation.
            kwargs: The keyword arguments

        Returns:
            An async iterable of ChatMessageContent.
        """
        if not isinstance(agent, ChatHistoryAgentProtocol):
            id = getattr(agent, "id", "")
            raise ServiceInvalidTypeError(
                f"Invalid channel binding for agent with id: `{id}` with name: ({type(agent).__name__})"
            )

        message_count = len(self.messages)

        async for response_message in agent.invoke_stream(self):
            if response_message.content:
                yield response_message

        for message_index in range(message_count, len(self.messages)):
            messages.append(self.messages[message_index])

    def _is_message_visible(self, message: ChatMessageContent) -> bool:
        """Determine if a message is visible to the user."""
        return not any(isinstance(item, (FunctionCallContent, FunctionResultContent)) for item in message.items)

    @override
    async def receive(
        self,
        history: list[ChatMessageContent],
    ) -> None:
        """Receive the conversation messages.

        Do not include messages that only contain file references.

        Args:
            history: The history of messages in the conversation.
        """
        filtered_history: list[ChatMessageContent] = []
        for message in history:
            new_message = deepcopy(message)
            if new_message.items is None:
                new_message.items = []
            allowed_items = [item for item in new_message.items if isinstance(item, self.ALLOWED_CONTENT_TYPES)]
            if not allowed_items:
                continue
            new_message.items.clear()
            new_message.items.extend(allowed_items)
            filtered_history.append(new_message)
        self.messages.extend(filtered_history)

    @override
    async def get_history(  # type: ignore
        self,
    ) -> AsyncIterable[ChatMessageContent]:
        """Retrieve the message history specific to this channel.

        Returns:
            An async iterable of ChatMessageContent.
        """
        for message in reversed(self.messages):
            yield message

    @override
    async def reset(self) -> None:
        """Reset the channel state."""
        self.messages.clear()
