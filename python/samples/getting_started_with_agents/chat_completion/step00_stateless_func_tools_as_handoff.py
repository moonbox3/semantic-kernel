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


@function_tool(description="Get the current financial market data.")
async def get_financial_data(ticker: str) -> str:
    print(f"[TOOL INVOKE] Fetching financial data for ticker: {ticker}")
    return f"The current financial data for {ticker} is stable."


weather_agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(),
    name="WeatherAgent",
    instructions="Use the provided tools, as required, to help answer the user's questions.",
    tools=[get_weather, reverse_string],
)

finance_agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(),
    name="FinanceAgent",
    instructions="Use the provided tools, as required, to help answer the user's questions.",
    tools=[get_financial_data],
)

manager_agent = ChatCompletionAgent(
    service=OpenAIChatCompletion(),
    name="ManagerAgent",
    instructions="Direct the user's request to the appropriate agent and/or tool.",
    tools=[weather_agent, finance_agent],
)


async def main():
    USER_INPUTS = [
        "What is the weather in New York?",
        "Get the current weather in Seattle and then reverse that word.",
        "Get the financial data for Apple.",
        "What is the weather in London?",
    ]

    for user_input in USER_INPUTS:
        print(f"User: {user_input}")
        response = await manager_agent.get_response(
            user_input,
        )
        print(f"{manager_agent.name}: {response}")
        print()

    """
    Sample Output:

    User: What is the weather in New York?
    [TOOL INVOKE] Fetching weather for city: New York
    ManagerAgent: The weather in New York is currently rainy. If you need more details, such as temperature or forecast,
        let me know!

    User: Get the current weather in Seattle and then reverse that word.
    [TOOL INVOKE] Fetching weather for city: Seattle
    ManagerAgent: The current weather in Seattle is rainy. The word "rainy" reversed is "yniar."

    User: Get the financial data for Apple.
    [TOOL INVOKE] Fetching financial data for ticker: AAPL
    ManagerAgent: Apple Inc. (AAPL) currently has a stable financial position. If you need detailed information such as 
        the current share price, recent financial performance, or specific financial ratios, please specify your request.

    User: What is the weather in London?
    [TOOL INVOKE] Fetching weather for city: London
    ManagerAgent: The weather in London is currently rainy. If you need more details like the temperature or a forecast,
        just let me know!
    """


if __name__ == "__main__":
    asyncio.run(main())
