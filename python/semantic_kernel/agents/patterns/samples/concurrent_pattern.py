# Copyright (c) Microsoft. All rights reserved.

import asyncio

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.patterns.core.concurrent import ConcurrentPattern
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion


async def main():
    """Main function to run the agents."""
    physics_agent = ChatCompletionAgent(
        name="PhysicsExpert",
        description="An expert in physics",
        instructions="You are an expert in physics.",
        service=OpenAIChatCompletion(),
    )
    chemistry_agent = ChatCompletionAgent(
        name="ChemistryExpert",
        description="An expert in chemistry",
        instructions="You are an expert in chemistry.",
        service=OpenAIChatCompletion(),
    )

    concurrent_pattern = ConcurrentPattern(agents=[physics_agent, chemistry_agent])
    await concurrent_pattern.start(task="Why is the sky blue in one sentence?", runtime=SingleThreadedAgentRuntime())


if __name__ == "__main__":
    asyncio.run(main())
