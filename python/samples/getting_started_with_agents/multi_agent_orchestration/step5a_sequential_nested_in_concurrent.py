# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.concurrent import (
    ConcurrentOrchestration,
    ConcurrentRequestMessage,
    ConcurrentResponseMessage,
)
from semantic_kernel.agents.orchestration.sequential import (
    SequentialOrchestration,
    SequentialRequestMessage,
    SequentialResultMessage,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.concurrent").setLevel(
    logging.DEBUG
)  # Enable DEBUG for the concurrent pattern
logging.getLogger("semantic_kernel.agents.orchestration.sequential").setLevel(
    logging.DEBUG
)  # Enable DEBUG for the sequential pattern


def get_sequential_orchestration(expert_agent: Agent, target_language: str) -> SequentialOrchestration:
    """Return a sequential orchestration that will be nested inside a concurrent orchestration."""
    translation_agent = ChatCompletionAgent(
        name="TranslationAgent",
        description="An agent that translates text to a target language.",
        instructions=f"Translate the text to {target_language}.",
        service=OpenAIChatCompletion(),
    )

    # Since the sequential orchestration will be nested inside the concurrent orchestration,
    # we need to transition the input message type from ConcurrentRequestMessage to the input
    # message type of the sequential orchestration.
    # We also we need to trasnsition the output of the sequential orchestration to the output
    # of the concurrent actor.
    async def input_transition_func(concurrent_input: ConcurrentRequestMessage) -> SequentialRequestMessage:
        # Convert the concurrent input to a sequential input message.
        return SequentialRequestMessage(
            body=concurrent_input.body,
        )

    async def output_transition_func(sequential_output: SequentialResultMessage) -> ConcurrentResponseMessage:
        # Convert the sequential output to a concurrent response message.
        return ConcurrentResponseMessage(
            body=sequential_output.body,
        )

    return SequentialOrchestration[ConcurrentRequestMessage, ConcurrentResponseMessage](
        workers=[expert_agent, translation_agent],
        external_input_message_type=ConcurrentRequestMessage,
        input_transition=input_transition_func,
        output_transition=output_transition_func,
    )


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
    chinese_physics_agent = ChatCompletionAgent(
        name="ChinesePhysicsExpert",
        description="An expert in physics who speaks Chinese.",
        instructions="You are an expert in physics. Answer in Chinese.",
        service=OpenAIChatCompletion(),
    )

    physics_sequential = get_sequential_orchestration(physics_agent, target_language="French")
    chemistry_sequential = get_sequential_orchestration(chemistry_agent, target_language="Spanish")

    concurrent_pattern = ConcurrentOrchestration(
        workers=[
            physics_sequential,
            chemistry_sequential,
            chinese_physics_agent,
        ]
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    result = await concurrent_pattern.invoke(
        task="Why is the sky blue in one sentence?",
        runtime=runtime,
    )

    await runtime.stop_when_idle()

    for agent_name, response in result.body.items():
        print(f"{agent_name} response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
