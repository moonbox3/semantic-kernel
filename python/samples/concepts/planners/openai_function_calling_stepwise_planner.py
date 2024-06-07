# Copyright (c) Microsoft. All rights reserved.

import asyncio
import datetime
import os

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.core_plugins import MathPlugin, SessionsPythonTool, TimePlugin
from semantic_kernel.exceptions.function_exceptions import FunctionExecutionException
from semantic_kernel.planners import FunctionCallingStepwisePlanner, FunctionCallingStepwisePlannerOptions

auth_token: AccessToken | None = None

ACA_TOKEN_ENDPOINT: str = "https://acasessions.io/.default"  # nosec


async def auth_callback() -> str:
    """Auth callback for the SessionsPythonTool.
    This is a sample auth callback that shows how to use Azure's DefaultAzureCredential
    to get an access token.
    """
    global auth_token
    current_utc_timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())

    if not auth_token or auth_token.expires_on < current_utc_timestamp:
        credential = DefaultAzureCredential()

        try:
            auth_token = credential.get_token(ACA_TOKEN_ENDPOINT)
        except ClientAuthenticationError as cae:
            err_messages = getattr(cae, "messages", [])
            raise FunctionExecutionException(
                f"Failed to retrieve the client auth token with messages: {' '.join(err_messages)}"
            ) from cae

    return auth_token.token


async def main():
    kernel = Kernel()

    service_id = "planner"
    kernel.add_service(
        OpenAIChatCompletion(
            service_id=service_id,
        ),
    )

    plugin_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 
        "resources", 
    )
    # cur_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")
    kernel.add_plugin(parent_directory=plugin_path, plugin_name="email_plugin")
    kernel.add_plugins(
        {
            "MathPlugin": MathPlugin(), 
            "TimePlugin": TimePlugin(), 
            "SessionsPythonTool": SessionsPythonTool(auth_callback=auth_callback)
        }
    )

    questions = [
        "Write a limerick, translate it to Spanish, and send it to Jane",
        "Compute the first 10 fibonacci numbers, run the code in Python, and send the result to John.",
    ]

    options = FunctionCallingStepwisePlannerOptions(
        max_iterations=10,
        max_tokens=4000,
    )

    planner = FunctionCallingStepwisePlanner(service_id=service_id, options=options)

    question_num = 1
    for question in questions:
        print(f"Working on question {question_num}...")
        result = await planner.invoke(kernel, question)
        print(f"Q: {question}\nA: {result.final_answer}\n")

        # Uncomment the following line to view the planner's process for completing the request
        print(f"\nChat history: {result.chat_history}\n")
        question_num += 1


if __name__ == "__main__":
    asyncio.run(main())
