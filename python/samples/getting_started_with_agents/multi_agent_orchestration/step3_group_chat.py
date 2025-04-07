# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.group_chat import GroupChatOrchestration, KernelFunctionGroupChatManager
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.kernel import Kernel

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.group_chat").setLevel(
    logging.DEBUG
)  # Enable DEBUG for group chat pattern


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

    async def user_input_function(chat_history: ChatHistory) -> str:
        """Function to get user input.

        This function simply finds the latest message from either the writer or editor agent and ask for user input.
        """
        for message in chat_history.messages[::-1]:
            if message.name == writer_agent.name:
                print(f"Here is the draft from {writer_agent.name}:")
                print(message.content)
                return input("What do you think? > ")

        return "No messages from writer agent found."

    kernel: Kernel = Kernel(services=[OpenAIChatCompletion()])
    group_chat_pattern = GroupChatOrchestration(
        manager=KernelFunctionGroupChatManager(kernel=kernel, max_rounds=10, user_input_func=user_input_function),
        agents=[
            writer_agent,
            editor_agent,
        ],
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    result = await group_chat_pattern.invoke(
        task="Please write a short story about the gingerbread man.",
        runtime=runtime,
    )

    await runtime.stop_when_idle()

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
