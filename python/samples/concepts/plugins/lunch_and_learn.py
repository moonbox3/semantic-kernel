# Copyright (c) Microsoft. All rights reserved.

# region Imports

import asyncio
import os
from functools import reduce
from typing import TYPE_CHECKING, Annotated

import pandas as pd

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_call_behavior import FunctionCallBehavior
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function

if TYPE_CHECKING:
    from semantic_kernel.functions import KernelFunction

# endregion

# region Custom Plugin


class Winner:
    """A data class to represent the winner of the giveaway."""
    def __init__(self, name: str, attendance: str, giveaway_type: str):
        self.name = name
        self.attendance = attendance
        self.giveaway_type = giveaway_type

    def __str__(self):
        return f"{self.name} ({self.giveaway_type})"


class SurprisePlugin:
    def __init__(self):
        self.past_winners = set()

    @kernel_function(
        name="run_giveaway", 
        description="Run a giveaway for the attendees."
    )
    def run_giveaway(
        self, 
        giveaway_type: Annotated[str, "The type of giveaway to run"],
        number_of_winners: Annotated[int, "The number of winners to select"],
    ) -> Annotated[list[Winner], "A list of Winner objects of the winners"]:
        """Run the giveaway by choosing the winners from the attendees list."""
        attendees = pd.read_excel(
            os.path.join(
                os.path.dirname(__file__), 
                "Attendee Information Form(1-47).xlsx",
            )
        )

        # Extract names of past winners without the giveaway type for filtering
        past_winner_names = {winner.split(" (")[0] for winner in self.past_winners}

        # Filter out past winners
        available_attendees = attendees[~attendees["Name2"].isin(past_winner_names)]
        
        if len(available_attendees) < number_of_winners:
            raise FunctionExecutionException("Not enough available attendees to select the requested number of winners.")  # noqa: E501
        
        winners = available_attendees.sample(n=number_of_winners)

        # Update the past winners set
        for _, row in winners.iterrows():
            self.past_winners.add(f"{row['Name2']} ({giveaway_type})")

        return [
            Winner(name=row["Name2"], attendance=row["How are you attending?"], giveaway_type=giveaway_type)
            for _, row in winners.iterrows()
        ]
    
    @kernel_function(
        name="get_past_winners",
        description="Get the list of past winners."
    )
    def get_past_winners(self) -> Annotated[list[str], "A list of strings of the past winners"]:
        """Get the list of past winners."""
        return list(self.past_winners)

# endregion

# region System Message


system_message = """
You are a chat bot. You are here to help people with their needs.
You are very helpful with a presentation related to the New York City
Engineering and Research Community. Your job is to help rally the attendees
with a cheerful attitude. You will help run a giveaway for the attendees.
"""

# endregion

# region SK Code

kernel = Kernel()

# Note: the underlying gpt-35/gpt-4 model version needs to be at least version 0613 to support tools.
kernel.add_service(OpenAIChatCompletion(service_id="chat"))

kernel.add_plugin(plugin=SurprisePlugin(), plugin_name="surprise")

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)

execution_settings = OpenAIChatPromptExecutionSettings(
    service_id="chat",
    max_tokens=4000,
    temperature=0.8,
    top_p=0.8,
    function_call_behavior=FunctionCallBehavior.EnableFunctions(
        auto_invoke=True, filters={"included_plugins": ["surprise"]}
    ),
)

history = ChatHistory()

history.add_system_message(system_message)

arguments = KernelArguments(settings=execution_settings)


def print_tool_calls(message: ChatMessageContent) -> None:
    # A helper method to pretty print the tool calls from the message.
    # This is only triggered if auto invoke tool calls is disabled.
    items = message.items
    formatted_tool_calls = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, FunctionCallContent):
            tool_call_id = item.id
            function_name = item.name
            function_arguments = item.arguments
            formatted_str = (
                f"tool_call {i} id: {tool_call_id}\n"
                f"tool_call {i} function name: {function_name}\n"
                f"tool_call {i} arguments: {function_arguments}"
            )
            formatted_tool_calls.append(formatted_str)
    print("Tool calls:\n" + "\n\n".join(formatted_tool_calls))


async def handle_streaming(
    kernel: Kernel,
    chat_function: "KernelFunction",
    arguments: KernelArguments,
) -> None:
    response = kernel.invoke_stream(
        chat_function,
        return_function_results=False,
        arguments=arguments,
    )

    print("Agent:> ", end="")
    streamed_chunks: list[StreamingChatMessageContent] = []
    async for message in response:
        if not execution_settings.function_call_behavior.auto_invoke_kernel_functions and isinstance(
            message[0], StreamingChatMessageContent
        ):
            streamed_chunks.append(message[0])
        else:
            print(str(message[0]), end="")

    if streamed_chunks:
        streaming_chat_message = reduce(lambda first, second: first + second, streamed_chunks)
        print("Auto tool calls is disabled, printing returned tool calls...")
        print_tool_calls(streaming_chat_message)

    print("\n")


async def chat() -> bool:
    try:
        user_input = input("User:> ")
    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False
    except EOFError:
        print("\n\nExiting chat...")
        return False

    if user_input == "exit":
        print("\n\nExiting chat...")
        return False
    arguments["user_input"] = user_input
    arguments["chat_history"] = history

    stream = True
    if stream:
        await handle_streaming(kernel, chat_function, arguments=arguments)
    else:
        result = await kernel.invoke(chat_function, arguments=arguments)

        # If tools are used, and auto invoke tool calls is False, the response will be of type
        # ChatMessageContent with information about the tool calls, which need to be sent
        # back to the model to get the final response.
        function_calls = [item for item in result.value[-1].items if isinstance(item, FunctionCallContent)]
        if not execution_settings.function_call_behavior.auto_invoke_kernel_functions and len(function_calls) > 0:
            print_tool_calls(result.value[0])
            return True

        print(f"Agent:> {result}")
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to the chat bot!\
        \n  Type 'exit' to exit.\
        \n  How can I help you today?"
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())

# endregion
