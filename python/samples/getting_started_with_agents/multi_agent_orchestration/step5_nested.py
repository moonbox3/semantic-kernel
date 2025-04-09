# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.concurrent import ConcurrentOrchestration
from semantic_kernel.agents.orchestration.sequential import SequentialOrchestration
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.concurrent").setLevel(
    logging.DEBUG
)  # Enable DEBUG for concurrent pattern


def get_sequential_orchestration(expert_agent: Agent, target_language: str) -> SequentialOrchestration:
    translation_agent = ChatCompletionAgent(
        name="TranslationAgent",
        description="An agent that translates text to a target language.",
        instructions=f"Translate the text to {target_language}.",
        service=OpenAIChatCompletion(),
    )

    return SequentialOrchestration(agents=[expert_agent, translation_agent])


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

    physics_sequential = get_sequential_orchestration(physics_agent, target_language="French")
    chemistry_sequential = get_sequential_orchestration(chemistry_agent, target_language="Spanish")

    concurrent_pattern = ConcurrentOrchestration(
        agents=[
            physics_sequential,
            chemistry_sequential,
        ]
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    result = await concurrent_pattern.invoke(
        task="Why is the sky blue in one sentence?",
        runtime=runtime,
    )

    await runtime.stop_when_idle()

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
