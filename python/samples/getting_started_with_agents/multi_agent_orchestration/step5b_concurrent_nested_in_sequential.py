# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.concurrent import (
    ConcurrentOrchestration,
    ConcurrentRequestMessage,
    ConcurrentResultMessage,
)
from semantic_kernel.agents.orchestration.sequential import (
    SequentialOrchestration,
    SequentialRequestMessage,
    SequentialResultMessage,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.concurrent").setLevel(
    logging.DEBUG
)  # Enable DEBUG for the concurrent pattern
logging.getLogger("semantic_kernel.agents.orchestration.sequential").setLevel(
    logging.DEBUG
)  # Enable DEBUG for the sequential pattern


def get_expert_concurrent_orchestration() -> ConcurrentOrchestration[SequentialRequestMessage, SequentialResultMessage]:
    """Return a concurrent orchestration that will be nested inside a sequential orchestration."""
    system_architect_agent = ChatCompletionAgent(
        name="SystemArchitectAgent",
        description="An agent that designs a system architecture.",
        instructions=(
            "You are a system architect. Given a task, try to approach it from a system architecture perspective. "
        ),
        service=OpenAIChatCompletion(),
    )
    database_expert_agent = ChatCompletionAgent(
        name="DatabaseExpertAgent",
        description="An agent that designs a database schema.",
        instructions=("You are a database expert. Given a task, try to approach it from a database perspective. "),
        service=OpenAIChatCompletion(),
    )
    security_expert_agent = ChatCompletionAgent(
        name="SecurityExpertAgent",
        description="An agent that designs a security architecture.",
        instructions=("You are a security expert. Given a task, try to approach it from a security perspective. "),
        service=OpenAIChatCompletion(),
    )
    devops_expert_agent = ChatCompletionAgent(
        name="DevOpsExpertAgent",
        description="An agent that designs a DevOps architecture.",
        instructions=("You are a DevOps expert. Given a task, try to approach it from a DevOps perspective. "),
        service=OpenAIChatCompletion(),
    )

    # Since the concurrent orchestration will be nested inside the sequential orchestration,
    # we need to transition the input message type from SequentialRequestMessage to the input
    # message type of the concurrent orchestration.
    # We also we need to trasnsition the output of the concurrent orchestration to the output
    # of a sequential actor, which is also a SequentialRequestMessage.
    def input_transition_func(sequential_input: SequentialRequestMessage) -> ConcurrentRequestMessage:
        # Convert the sequential input to a concurrent input message.
        return ConcurrentRequestMessage(
            body=sequential_input.body,
        )

    def output_transition_func(concurrent_output: ConcurrentResultMessage) -> SequentialRequestMessage:
        # Convert the concurrent output to a sequential response message.
        return SequentialRequestMessage(
            body=ChatMessageContent(
                role=AuthorRole.USER,
                content="\n".join([
                    f"{agent_name}: {response}" for agent_name, response in concurrent_output.body.items()
                ]),
            ),
        )

    return ConcurrentOrchestration(
        workers=[
            system_architect_agent,
            database_expert_agent,
            security_expert_agent,
            devops_expert_agent,
        ],
        external_input_message_type=SequentialRequestMessage,
        input_transition=input_transition_func,
        output_transition=output_transition_func,
    )


def get_mermaid_concurrent_orchestration() -> ConcurrentOrchestration[
    SequentialRequestMessage, SequentialResultMessage
]:
    """Return a concurrent orchestration that will be nested inside a sequential orchestration."""
    donet_mermain_agent_name = "DotnetMermaidAgent"
    dotnet_mermaid_agent = ChatCompletionAgent(
        name=donet_mermain_agent_name,
        description="An agent that generates a mermaid diagram spectifically for .NET.",
        instructions=("Given a system architecture, generate a class diagram in mermaid format for .NET."),
        service=OpenAIChatCompletion(),
    )

    python_mermaid_agent_name = "PythonMermaidAgent"
    python_mermaid_agent = ChatCompletionAgent(
        name=python_mermaid_agent_name,
        description="An agent that generates a mermaid diagram specifically for Python.",
        instructions=("Given a system architecture, generate a class diagram in mermaid format for Python."),
        service=OpenAIChatCompletion(),
    )

    # Since the concurrent orchestration will be nested inside the sequential orchestration,
    # we need to transition the input message type from SequentialRequestMessage to the input
    # message type of the concurrent orchestration.
    # We also we need to trasnsition the output of the concurrent orchestration to the output
    # of the sequential actor.
    def input_transition_func(sequential_input: SequentialRequestMessage) -> ConcurrentRequestMessage:
        # Convert the sequential input to a concurrent input message.
        return ConcurrentRequestMessage(
            body=sequential_input.body,
        )

    def output_transition_func(concurrent_output: ConcurrentResultMessage) -> SequentialRequestMessage:
        # Convert the concurrent output to a sequential response message.
        return SequentialRequestMessage(
            body=ChatMessageContent(
                role=AuthorRole.USER,
                content="\n".join([
                    f"{'.Net' if donet_mermain_agent_name in agent_name else 'Python'}: {response}"
                    for agent_name, response in concurrent_output.body.items()
                ]),
            ),
        )

    return ConcurrentOrchestration(
        workers=[dotnet_mermaid_agent, python_mermaid_agent],
        external_input_message_type=SequentialRequestMessage,
        input_transition=input_transition_func,
        output_transition=output_transition_func,
    )


async def main():
    """Main function to run the agents."""
    expert_concurrent_orchstration = get_expert_concurrent_orchestration()
    mermaid_concurrent_orchestration = get_mermaid_concurrent_orchestration()

    enginnering_lead = ChatCompletionAgent(
        name="EngineeringLeadAgent",
        description="An agent that reviews the system design.",
        instructions=(
            "You are an engineering lead. You will be presented with considerations from the experts in your team. "
            "Based on the considerations, design the system architecture."
        ),
        service=OpenAIChatCompletion(),
    )

    sequential_orchestration = SequentialOrchestration(
        workers=[
            expert_concurrent_orchstration,
            enginnering_lead,
            mermaid_concurrent_orchestration,
        ],
    )

    runtime = SingleThreadedAgentRuntime()
    runtime.start()

    result = await sequential_orchestration.invoke(
        task="Design a video streaming plaform.",
        runtime=runtime,
    )

    await runtime.stop_when_idle()

    print(result.body)


if __name__ == "__main__":
    asyncio.run(main())
