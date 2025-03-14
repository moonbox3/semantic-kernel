# Copyright (c) Microsoft. All rights reserved.


import asyncio

from autogen_core import SingleThreadedAgentRuntime, TopicId, TypeSubscription

from semantic_kernel.agents.patterns.group_chat.agents import (
    EditorAgent,
    GroupChatManager,
    IllustratorAgent,
    UserAgent,
    WriterAgent,
)
from semantic_kernel.agents.patterns.group_chat.message_types import GroupChatMessage
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


async def main():
    runtime = SingleThreadedAgentRuntime()

    await WriterAgent.register(runtime, "WriterAgent", lambda: WriterAgent())
    await runtime.add_subscription(TypeSubscription("WriterAgentTopic", "WriterAgent"))
    await runtime.add_subscription(TypeSubscription("GroupChatTopic", "WriterAgent"))

    await EditorAgent.register(runtime, "EditorAgent", lambda: EditorAgent())
    await runtime.add_subscription(TypeSubscription("EditorAgentTopic", "EditorAgent"))
    await runtime.add_subscription(TypeSubscription("GroupChatTopic", "EditorAgent"))

    await IllustratorAgent.register(runtime, "IllustratorAgent", lambda: IllustratorAgent())
    await runtime.add_subscription(TypeSubscription("IllustratorAgentTopic", "IllustratorAgent"))
    await runtime.add_subscription(TypeSubscription("GroupChatTopic", "IllustratorAgent"))

    await UserAgent.register(runtime, "UserAgent", lambda: UserAgent())
    await runtime.add_subscription(TypeSubscription("UserAgentTopic", "UserAgent"))
    await runtime.add_subscription(TypeSubscription("GroupChatTopic", "UserAgent"))

    await GroupChatManager.register(
        runtime,
        "GroupChatManager",
        lambda: GroupChatManager(
            participant_descriptions={
                "WriterAgent": WriterAgent.description(),
                "EditorAgent": EditorAgent.description(),
                "IllustratorAgent": IllustratorAgent.description(),
                "UserAgent": UserAgent.description(),
            },
            participant_topics={
                "WriterAgent": "WriterAgentTopic",
                "EditorAgent": "EditorAgentTopic",
                "IllustratorAgent": "IllustratorAgentTopic",
                "UserAgent": "UserAgentTopic",
            },
        ),
    )
    await runtime.add_subscription(TypeSubscription("GroupChatTopic", "GroupChatManager"))

    runtime.start()
    await runtime.publish_message(
        GroupChatMessage(
            body=ChatMessageContent(
                AuthorRole.USER,
                content=(
                    "Please write a short story about the gingerbread man with up to 3 photo-realistic illustrations."
                ),
                name="User",
            )
        ),
        topic_id=TopicId("GroupChatTopic", "default"),
    )
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
