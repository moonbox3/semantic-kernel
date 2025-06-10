# Copyright (c) Microsoft. All rights reserved.

import asyncio

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.filters import AutoFunctionInvocationContext, FilterTypes
from semantic_kernel.filters.agent_contexts import PromptRenderContext
from semantic_kernel.functions.function_tools import function_tool

"""
This sample demonstrates how to use filters with a ChatCompletionAgent.
Filters can be used to modify the agent's behavior at various stages, such as during prompt rendering,
function invocation, and auto function invocation.

In this sample a prompt rendering filter is used to modify the system message sent to the agent, and 
an auto function invocation filter is used to intercept a function invocation and return a custom message.
"""


@function_tool(description="Get the weather for a given city.")
async def get_weather(city: str, location: str | None = None) -> str:
    print(f"[TOOL INVOKE] Fetching weather for city: {city}")
    return f"The weather for location {location} in {city} is rainy."


agent = ChatCompletionAgent(
    name="mosscap",
    description="Chat bot that figures out what people need, but hates math",
    service=OpenAIChatCompletion(),
    tools=[get_weather],
)


# attach a prompt filter directly to the agent
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


async def main():
    user_input = "What is the weather in Seattle?"
    print(f"User input: {user_input}")
    response = await agent.get_response(user_input)
    if isinstance(response.content.items[0], FunctionResultContent):
        print(f"Agent: {response.content.items[0].result}")
    else:
        print(f"Agent: {response.content}")

    """
    Sample output:

    If `context.terminate` is True:

    User input: What is the weather in Seattle?
    <Auto Function Invocation filter suppresses the weather tool>
    Agent: Stop asking me the weather. It's sunny.

    If `context.terminate` is False:

    User input: What is the weather in Seattle?
    <Prompt filter tells agent to answer in a silly way>
    Agent: The weather in Seattle is currently... SUNNY! 
        Yes, you read that right—sunny! Birds are surfing on rainbows, 
        people are grilling pineapples on their umbrellas, and rainclouds 
        are on vacation in Florida. Grab your sunglasses and go soak up 
        that mythical Seattle sunshine while it's still legal!
    """


if __name__ == "__main__":
    asyncio.run(main())
