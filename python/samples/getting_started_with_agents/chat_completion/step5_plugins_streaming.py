# Copyright (c) Microsoft. All rights reserved.

import asyncio
from typing import Annotated

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function

###################################################################
# The following sample demonstrates how to create a simple,       #
# non-group agent that utilizes plugins defined as part of        #
# the Kernel.                                                     #
###################################################################


# Define a sample plugin for the sample
class MenuPlugin:
    """A sample Menu Plugin used for the concept sample."""

    @kernel_function(description="Provides a list of specials from the menu.")
    def get_specials(self) -> str:
        return """
        Special Soup: Clam Chowder
        Special Salad: Cobb Salad
        Special Drink: Chai Tea
        """

    @kernel_function(description="Provides the price of the requested menu item.")
    def get_item_price(self, menu_item: Annotated[str, "The name of the menu item."]) -> str:
        return "$9.99"

    @kernel_function(description="Provides the hours of the restaurant.")
    def get_hours(self) -> str:
        return "We are open from 9am to 9pm every day."


async def main():
    agent = ChatCompletionAgent(
        service=OpenAIChatCompletion(),
        plugins=[MenuPlugin()],
        instructions="You are a helpful assistant that can answer questions about the menu",
        auto_function_calling=True,  # The flag will enable FunctionChoiceBehavior.Auto() during the invoke
    )
    user_inputs = [
        "What is the special soup?",
        "What hours are you open?",
    ]

    for input in user_inputs:
        print(f"# User Input: {input}")
        async for response in agent.invoke_stream(input):
            print(response.content, end="", flush=True)
        print()


if __name__ == "__main__":
    asyncio.run(main())
