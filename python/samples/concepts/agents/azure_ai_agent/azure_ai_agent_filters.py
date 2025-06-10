# Copyright (c) Microsoft. All rights reserved.

import asyncio

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.filters.agent_contexts import AutoFunctionInvocationContext, PromptRenderContext
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.functions.function_tools import function_tool
from semantic_kernel.functions.kernel_arguments import KernelArguments

"""
The following sample demonstrates how to create an Azure AI agent that answers
questions about a sample menu using a Semantic Kernel Plugin.
"""


# Define a sample plugin for the sample
@function_tool(description="Get the weather for a given city.")
async def get_weather(city: str, location: str | None = None) -> str:
    print(f"[TOOL INVOKE] Fetching weather for city: {city}, and location: {location}")
    return f"The weather for location {location} in {city} is rainy."


@function_tool(description="Reverse a string.")
async def reverse_string(text: str) -> str:
    """Reverses the input string."""
    print(f"[TOOL INVOKE] Reversing the string: {text}")
    return text[::-1]


# Simulate a conversation with the agent
USER_INPUTS = [
    "What is the weather in New York?",
    "Get the current weather in Seattle and then reverse that word.",
    "What is the weather in Seattle?",
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
            arguments=KernelArguments(location="USA"),
        )

        @agent.filter(FilterTypes.PROMPT_RENDERING)
        async def prompt_rendering_filter(context: PromptRenderContext, next):
            await next(context)
            context.rendered_prompt = f"You should answer in the silliest way possible {context.rendered_prompt or ''}"  # noqa: E501

        # attach an auto-function filter directly to the agent
        @agent.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def suppress_weather(context: AutoFunctionInvocationContext, next):
            if context.function.name == "get_weather":
                # This will suppress the weather tool invocation
                # and return a custom refusal message
                context.function_result = "Stop asking me the weather. It's sunny."
                # TOGGLE this to True if you want to stop the agent from continuing
                context.terminate = False
            else:
                await next(context)

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
        [TOOL INVOKE] Fetching weather for city: New York, and location: USA
        # WeatherAgent: The weather in New York is rainy. Is there anything else you would like to know about the 
            weather?

        # User: Get the current weather in Seattle and then reverse that word.
        [TOOL INVOKE] Fetching weather for city: Seattle, and location: USA
        [TOOL INVOKE] Reversing the string: Seattle
        # WeatherAgent: The current weather in Seattle is rainy. Also, the word "Seattle" reversed is "elttaeS". 
            If you need more information, feel free to ask!

        # User: What is the weather in Seattle?
        # WeatherAgent: The weather in Seattle is rainy. If you would like to know more details or the forecast for 
            another city, just let me know!
        """


if __name__ == "__main__":
    asyncio.run(main())
