# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from typing import Annotated

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.patterns.core.group_chat import GroupChatPattern, KernelFunctionGroupChatManager
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel

logging.basicConfig(level=logging.WARNING)  # Set default level to INFO
logging.getLogger("semantic_kernel.agents.patterns.core.group_chat").setLevel(
    logging.DEBUG
)  # Enable DEBUG for group chat pattern


class UserPlugin:
    """A plugin that interacts with the user."""

    @kernel_function(description="Interact with the user.")
    def interact(
        self, text: Annotated[str, "The text to be displayed to the user."]
    ) -> Annotated[str, "User's response."]:
        """Interact with the user."""
        return input(f"{text}\n> ")


async def main():
    """Main function to run the agents."""
    writer_agent = ChatCompletionAgent(
        name="WriterAgent",
        description="Writer for creating any text content.",
        instructions=("You are a Writer. You produce good work."),
        service=OpenAIChatCompletion(),
    )
    editor_agent = ChatCompletionAgent(
        name="EditorAgent",
        description="Editor for planning and reviewing the content.",
        instructions=(
            "You are an Editor. Plan and guide the task given by the user. Provide critical feedbacks to the draft. "
            "Approve if the task is completed and the draft meets user's requirements."
        ),
        service=OpenAIChatCompletion(),
    )
    user_proxy = ChatCompletionAgent(
        name="UserProxy",
        description="A user proxy to show user the draft and get feedback.",
        instructions=(
            "You are a user proxy. You interact with the user and summarize the user feedback to the writer and editor."
            " You are responsible for communicating the user's requirements to the writer and editor."
        ),
        service=OpenAIChatCompletion(),
        plugins=[UserPlugin()],
    )

    kernel: Kernel = Kernel(services=[OpenAIChatCompletion()])

    group_chat_pattern = GroupChatPattern(
        manager=KernelFunctionGroupChatManager(kernel=kernel, max_rounds=10),
        agents=[
            writer_agent,
            editor_agent,
            user_proxy,
        ],
    )
    await group_chat_pattern.start(
        task="Please write a short story about the gingerbread man.",
        runtime=SingleThreadedAgentRuntime(),
    )


if __name__ == "__main__":
    asyncio.run(main())
