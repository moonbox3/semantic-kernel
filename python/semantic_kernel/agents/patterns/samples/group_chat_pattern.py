# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.patterns.core.group_chat import GroupChatPattern, KernelFunctionGroupChatManager
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.kernel import Kernel


async def main():
    """Main function to run the agents."""
    writer_agent = ChatCompletionAgent(
        name="WriterAgent",
        description="Writer for creating any text content.",
        instructions=(
            "You are an Illustrator. You use the generate_image tool to create images given user's requirement. "
            "Make sure the images have consistent characters and style."
        ),
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

    kernel: Kernel = Kernel(services=[OpenAIChatCompletion()])

    group_chat_pattern = await GroupChatPattern.create(
        manager=KernelFunctionGroupChatManager(kernel=kernel, max_rounds=5),
        agents=[
            writer_agent,
            editor_agent,
        ],
    )
    await group_chat_pattern.start(task="Please write a short story about the gingerbread man.")


if __name__ == "__main__":
    asyncio.run(main())
