# Copyright (c) Microsoft. All rights reserved.

import asyncio

from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


async def main():
    """Main function to run the agents."""
    runtime = SingleThreadedAgentRuntime()

    await ChatCompletionAgent.register(
        runtime,
        "agent_1",
        lambda: ChatCompletionAgent(
            name="Agent1",
            description="test",
            service=OpenAIChatCompletion(),
        ),
    )
    await ChatCompletionAgent.register(
        runtime,
        "agent_2",
        lambda: ChatCompletionAgent(
            name="Agent2",
            description="test",
            service=OpenAIChatCompletion(),
        ),
    )

    message = ChatMessageContent(AuthorRole.USER, content="Why is the sky blue in one sentence?")

    runtime.start()
    await runtime.publish_message(message, topic_id=DefaultTopicId())
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
