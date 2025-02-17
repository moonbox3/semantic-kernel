# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

"""
The following sample demonstrates how to create a simple, Chat Completion 
agent that responds to user input using the Azure Chat Completion service
in a streaming fashion.
"""


async def main():
    agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        instructions="You are a helpful assistant that can answer questions about the world",
    )
    user_inputs = (
        "Why is the sky blue?",
        "How do clouds form?",
    )

    for input in user_inputs:
        print(f"# User Input: {input}")
        async for response in agent.invoke_stream(input):
            print(response.content, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
