# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import os.path
from typing import Annotated

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function

# Determine the directory of the current script
script_dir = os.path.dirname(__file__)

# Construct the full path to the storage directory
PERSIST_DIR = os.path.join(script_dir, "storage")


#########################################################################
# Step 0: Create the LlamaIndexPlugin that aids in performing RAG.    ###
# This uses the `llama-index` package, so please install if necessary.###
#
# It is using some sample Semantic Kernel PDF documents in a `data`   ###
# directory to create the index.                                      ###
#########################################################################


class LLamaIndexPlugin:
    """
    This class is a plugin that uses the LLamaIndex to perform simple RAG.

    It's using a simple `VectorStoreIndex` to index the documents and a `BaseQueryEngine` to query them.
    """

    def __init__(self, persist_index_to_disk: bool = True):
        """
        Initialize a new instance of the LLamaIndexPlugin.

        Args:
            persist_index_to_disk (bool): Whether to persist the index to disk. Defaults to True.
        """
        self.persist_index_to_disk = persist_index_to_disk
        self.index = self.initialize_index()
        self.query_engine = self.index.as_query_engine()

    def initialize_index(self):
        """
        Initialize the index either by loading existing data from disk or creating a new one.

        Returns:
            index (VectorStoreIndex): The initialized index.
        """
        if not os.path.exists(PERSIST_DIR):
            return self.create_new_index()
        return self.load_existing_index()

    def create_new_index(self):
        """
        Create a new index from documents in the data directory.

        Returns:
            index (VectorStoreIndex): The created index.
        """
        data_dir = os.path.join(script_dir, "data")
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        if self.persist_index_to_disk:
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        return index

    def load_existing_index(self):
        """
        Load the existing index from the storage directory.

        Returns:
            index (VectorStoreIndex): The loaded index.
        """
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)

    @kernel_function(
        name="query_documents", description="Query the documents in the index to answer the user's question."
    )
    def query_documents(self, question: Annotated[str, "The question to answer"]) -> str:
        """
        Query the documents in the index to answer the user's question.

        Args:
            question (str): The question to answer.

        Returns:
            str: The answer to the question.
        """
        return self.query_engine.query(question)


system_message = """
You are a chat bot. You help people find answers to their questions about Semantic Kernel.
You are to only respond based on the documents you have access to and you are not to make up
any information if you don't know it. It's okay to respond with "I don't know" if you can't find
the information. You can also ask for clarification if you need more information to answer a question.
"""

#########################################################################
# Step 1: Define the Semantic Kernel related pieces.                  ###
# This includes the kernel, services, plugins, and execution settings.###
#########################################################################

kernel = Kernel()

service_id = "docs-chat"
kernel.add_service(AzureChatCompletion(service_id=service_id))

kernel.add_plugin(LLamaIndexPlugin(), plugin_name="llama_index")

chat_function = kernel.add_function(
    prompt="{{$chat_history}}{{$user_input}}",
    plugin_name="ChatBot",
    function_name="Chat",
)

# We're using Automatic Function Calling for this example.
# This is configured by setting the `function_choice_behavior` to `Auto` or
# by specifying: `function_choice_behavior="auto"`. Note that if using the latter,
# the `filters` parameter is not included and thus all functions are considered.
execution_settings = AzureChatPromptExecutionSettings(
    service_id=service_id,
    max_tokens=2000,
    temperature=0.2,
    top_p=0.8,
    function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["ChatBot"]}),
)

history = ChatHistory()

history.add_system_message(system_message)

arguments = KernelArguments(settings=execution_settings)

################################################################
# Step 2: Define the chat loop used for user interaction.    ###
################################################################


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

    result = await kernel.invoke(chat_function, arguments=arguments)

    print(f"SK Docs Bot:> {result}")
    history.add_user_message(user_input)
    history.add_assistant_message(str(result))
    return True


async def main() -> None:
    chatting = True
    print(
        "Welcome to the Semantic Kernel docs chat bot!\
        \n  Type 'exit' to exit.\
        \n  Try asking questions to learn more about Semantic Kernel."
    )
    while chatting:
        chatting = await chat()


if __name__ == "__main__":
    asyncio.run(main())
