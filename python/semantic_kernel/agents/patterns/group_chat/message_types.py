# Copyright (c) Microsoft. All rights reserved.


from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.kernel_pydantic import KernelBaseModel


class GroupChatMessage(KernelBaseModel):
    body: ChatMessageContent


class RequestToSpeak(KernelBaseModel):
    pass
