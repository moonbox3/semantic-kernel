# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.function_tools import function_tool


@function_tool(description="Get the weather for a given city.")
async def get_weather(city: str) -> str:
    print(f"[TOOL INVOKE] Fetching weather for city: {city}")
    return f"The weather in {city} is rainy."


@function_tool(description="Reverse a string.")
async def reverse_string(text: str) -> str:
    """Reverses the input string."""
    print(f"[TOOL INVOKE] Reversing the string: {text}")
    return text[::-1]


weather_agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(),
    name="GreeterAgent",
    instructions="Use the provided tools, as required, to help answer the user's questions.",
    tools=[get_weather, reverse_string],
)


async def main():
    USER_INPUTS = [
        "What is the weather in New York?",
        "Get the current weather in Seattle and then reverse that word.",
        "What is the weather in London?",
    ]

    for user_input in USER_INPUTS:
        print(f"User: {user_input}")
        response = await weather_agent.get_response(
            user_input,
        )
        print(f"{weather_agent.name}: {response}")
        print()

    """
    Sample Output:

    User: What is the weather in New York?
    [TOOL INVOKE] Fetching weather for city: New York
    WeatherAgent: The weather in New York is currently rainy. If you need more details, feel free to ask!

    User: Get the current weather in Seattle and then reverse that word.
    [TOOL INVOKE] Fetching weather for city: Seattle
    [TOOL INVOKE] Reversing the string: Seattle
    WeatherAgent: The current weather in Seattle is rainy. Also, the word "Seattle" reversed is "elttaeS".

    User: What is the weather in London?
    [TOOL INVOKE] Fetching weather for city: London
    WeatherAgent: The weather in London is currently rainy. If you need more details like temperature or forecast, 
        let me know!
    """


if __name__ == "__main__":
    asyncio.run(main())
