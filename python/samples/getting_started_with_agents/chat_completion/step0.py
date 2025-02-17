# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class WeatherPlugin:
    """A sample class to get the weather."""

    @kernel_function
    def get_weather(self, city: str) -> str:
        if city.lower() == "paris":
            return f"The weather in {city} is cloudy today."
        return f"The weather in {city} is sunny today."


async def main():
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(),
        plugins=[WeatherPlugin()],
        instructions="you are a helpful assistant that can answer questions about the world",
        auto_function_calling=True,  # The flag will enable FunctionChoiceBehavior during the invoke
    )
    user_inputs = (
        "What is the current weather in New York City?",
        "What is the weather like in Paris?",
    )

    streaming = True

    if streaming:
        for input in user_inputs:
            async for response in agent.invoke_stream(input):
                print(response.content, end="", flush=True)
            print()
    else:
        for input in user_inputs:
            print(await agent.invoke(input))


if __name__ == "__main__":
    asyncio.run(main())
