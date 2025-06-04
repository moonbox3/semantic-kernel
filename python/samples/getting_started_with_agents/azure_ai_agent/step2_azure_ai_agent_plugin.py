# Copyright (c) Microsoft. All rights reserved.

import asyncio

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.functions.function_tools import function_tool
from semantic_kernel.functions.kernel_arguments import KernelArguments

"""
The following sample demonstrates how to create an Azure AI agent that answers
questions about a sample menu using a Semantic Kernel Plugin.
"""


# Define a sample plugin for the sample
@function_tool(description="Get the weather for a given city.")
async def get_weather(city: str) -> str:
    print(f"[TOOL INVOKE] Fetching weather for city: {city}")
    return f"The weather in {city} is rainy."


@function_tool(description="Reverse a string.")
async def reverse_string(text: str) -> str:
    """Reverses the input string."""
    print(f"[TOOL INVOKE] Reversing the string: {text}")
    return text[::-1]


# Simulate a conversation with the agent
USER_INPUTS = [
    "What is the weather in New York?",
    "Get the current weather in Seattle and then reverse that word.",
    "What is the weather in London?",
]


async def main() -> None:
    async with (
        DefaultAzureCredential() as creds,
        AzureAIAgent.create_client(credential=creds) as client,
    ):
        # 1. Create an agent on the Azure AI agent service
        agent_definition = await client.agents.create_agent(
            model=AzureAIAgentSettings().model_deployment_name,
            name="WeatherAgent",
            instructions="Answer questions about the weather.",
        )

        # 2. Create a Semantic Kernel agent for the Azure AI agent
        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
            tools=[get_weather, reverse_string],
            arguments=KernelArguments(style="funny"),
        )

        # 3. Create a thread for the agent
        # If no thread is provided, a new thread will be
        # created and returned with the initial response
        thread = None

        try:
            for user_input in USER_INPUTS:
                print(f"# User: {user_input}")
                # 4. Invoke the agent for the specified thread for response
                async for response in agent.invoke(
                    messages=user_input,
                    thread=thread,
                ):
                    print(f"# {response.name}: {response}")
                    thread = response.thread
                print()
        finally:
            # 5. Cleanup: Delete the thread and agent
            await thread.delete() if thread else None
            await client.agents.delete_agent(agent.id)

        """
        Sample Output:

        # User: What is the weather in New York?
        [TOOL INVOKE] Fetching weather for city: New York
        # WeatherAgent: The weather in New York is currently rainy.

        # User: Get the current weather in Seattle and then reverse that word.
        [TOOL INVOKE] Fetching weather for city: Seattle
        [TOOL INVOKE] Reversing the string: rainy
        # WeatherAgent: The current weather in Seattle is rainy. Reversed, the word "rainy" is "yniar".

        # User: What is the weather in London?
        [TOOL INVOKE] Fetching weather for city: London
        # WeatherAgent: The weather in London is currently rainy.
        """


if __name__ == "__main__":
    asyncio.run(main())
