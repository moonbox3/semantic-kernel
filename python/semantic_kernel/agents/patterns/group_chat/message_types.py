# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class GroupChatMessage(KernelBaseModel):
    """A message in a group chat."""

    body: ChatMessageContent


class RequestToSpeak(KernelBaseModel):
    """A request to speak sent to participants in a group chat."""

    pass


class GroupChatReset(KernelBaseModel):
    """A message to reset a participant's chat history in a group chat."""

    pass
