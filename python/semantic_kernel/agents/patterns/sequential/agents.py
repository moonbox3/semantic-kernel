# Copyright (c) Microsoft. All rights reserved.

from autogen_core import MessageContext, RoutedAgent, message_handler

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent

CONCEPT_EXTRACTOR_TOPIC_TYPE = "ConceptExtractorAgent"
WRITER_TOPIC_TYPE = "WriterAgent"
FORMAT_PROOF_TOPIC_TYPE = "FormatProofAgent"
USER_TOPIC_TYPE = "User"


class ConceptExtractorAgent(ChatCompletionAgent):
    """Agent that extracts concepts from a product description."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(
            name="ConceptExtractorAgent",
            description="ConceptExtractorAgent",
            instructions=(
                "You are a marketing analyst. Given a product description, identify:\n"
                "- Key features\n"
                "- Target audience\n"
                "- Unique selling points\n\n"
            ),
            service=OpenAIChatCompletion(),
            publish_topics=[WRITER_TOPIC_TYPE],
        )


class WriterAgent(ChatCompletionAgent):
    """Agent that writes a marketing copy based on extracted concepts."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(
            name="WriterAgent",
            description="WriterAgent",
            instructions=(
                "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
                "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
                "Output should be short (around 150 words), output just the copy as a single text block."
            ),
            service=OpenAIChatCompletion(),
            publish_topics=[FORMAT_PROOF_TOPIC_TYPE],
        )


class FormatProofAgent(ChatCompletionAgent):
    """Agent that formats and proofreads the marketing copy."""

    def __init__(self):
        """Initialize the agent."""
        super().__init__(
            name="WriterAgent",
            description="WriterAgent",
            instructions=(
                "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
                "give format and make it polished. Output the final improved copy as a single text block."
            ),
            service=OpenAIChatCompletion(),
            publish_topics=[USER_TOPIC_TYPE],
        )


class UserAgent(RoutedAgent):
    """A user agent that outputs the final copy to the user."""

    def __init__(self) -> None:
        """Initialize the agent."""
        super().__init__("A user agent that outputs the final copy to the user.")

    @message_handler
    async def _handle_final_copy(self, message: ChatMessageContent, ctx: MessageContext) -> None:
        print(f"\n{'-' * 80}\n{self.id.type} received final copy:\n{message.content}")
