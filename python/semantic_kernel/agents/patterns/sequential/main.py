# Copyright (c) Microsoft. All rights reserved.


import asyncio

from autogen_core import SingleThreadedAgentRuntime, TopicId, TypeSubscription

from semantic_kernel.agents.patterns.sequential.agents import (
    CONCEPT_EXTRACTOR_TOPIC_TYPE,
    FORMAT_PROOF_TOPIC_TYPE,
    USER_TOPIC_TYPE,
    WRITER_TOPIC_TYPE,
    ConceptExtractorAgent,
    FormatProofAgent,
    UserAgent,
    WriterAgent,
)
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


async def main():
    runtime = SingleThreadedAgentRuntime()

    await ConceptExtractorAgent.register(
        runtime,
        "ConceptExtractorAgent",
        lambda: ConceptExtractorAgent(),
    )
    await WriterAgent.register(
        runtime,
        "WriterAgent",
        lambda: WriterAgent(),
    )
    await FormatProofAgent.register(
        runtime,
        "FormatProofAgent",
        lambda: FormatProofAgent(),
    )
    await UserAgent.register(
        runtime,
        "UserAgent",
        lambda: UserAgent(),
    )
    await runtime.add_subscription(
        TypeSubscription(topic_type=CONCEPT_EXTRACTOR_TOPIC_TYPE, agent_type="ConceptExtractorAgent")
    )
    await runtime.add_subscription(TypeSubscription(topic_type=WRITER_TOPIC_TYPE, agent_type="WriterAgent"))
    await runtime.add_subscription(TypeSubscription(topic_type=FORMAT_PROOF_TOPIC_TYPE, agent_type="FormatProofAgent"))
    await runtime.add_subscription(TypeSubscription(topic_type=USER_TOPIC_TYPE, agent_type="UserAgent"))

    message = ChatMessageContent(
        AuthorRole.USER,
        content="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours",
    )

    runtime.start()
    await runtime.publish_message(message, topic_id=TopicId(CONCEPT_EXTRACTOR_TOPIC_TYPE, "default"))
    await runtime.stop_when_idle()


if __name__ == "__main__":
    asyncio.run(main())
