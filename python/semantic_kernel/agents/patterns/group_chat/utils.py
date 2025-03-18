# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.image_content import ImageContent


def format_chat_history(chat_history: ChatHistory) -> str:
    """Format the chat history into a human-readable string."""
    messages: list[str] = []

    for message in chat_history.messages:
        formatted_message = ""
        if message.content:
            formatted_message = f"{message.role} ({message.name or ''}): {message.content}"
        for item in message.items:
            if isinstance(item, ImageContent):
                if formatted_message:
                    formatted_message += "\n[Image]"
                else:
                    formatted_message = f"{message.role} ({message.name or ''}): [Image]"

        if formatted_message:
            messages.append(formatted_message)

    return "\n".join(messages)
