# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.sequential import SequentialOrchestration
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion

logging.basicConfig(level=logging.WARNING)  # Set default level to WARNING
logging.getLogger("semantic_kernel.agents.orchestration.sequential").setLevel(
    logging.DEBUG
)  # Enable DEBUG for the sequential pattern


async def main():
    """Main function to run the agents."""
    concept_extractor_agent = ChatCompletionAgent(
        name="ConceptExtractorAgent",
        description="A agent that extracts key concepts from a product description.",
        instructions=(
            "You are a marketing analyst. Given a product description, identify:\n"
            "- Key features\n"
            "- Target audience\n"
            "- Unique selling points\n\n"
        ),
        service=OpenAIChatCompletion(),
    )
    writer_agent = ChatCompletionAgent(
        name="WriterAgent",
        description="An agent that writes a marketing copy based on the extracted concepts.",
        instructions=(
            "You are a marketing copywriter. Given a block of text describing features, audience, and USPs, "
            "compose a compelling marketing copy (like a newsletter section) that highlights these points. "
            "Output should be short (around 150 words), output just the copy as a single text block."
        ),
        service=OpenAIChatCompletion(),
    )
    format_proof_agent = ChatCompletionAgent(
        name="FormatProofAgent",
        description="An agent that formats and proofreads the marketing copy.",
        instructions=(
            "You are an editor. Given the draft copy, correct grammar, improve clarity, ensure consistent tone, "
            "give format and make it polished. Output the final improved copy as a single text block."
        ),
        service=OpenAIChatCompletion(),
    )

    sequential_pattern = SequentialOrchestration(
        agents=[
            concept_extractor_agent,
            writer_agent,
            format_proof_agent,
        ]
    )
    await sequential_pattern.start(
        task="An eco-friendly stainless steel water bottle that keeps drinks cold for 24 hours",
        runtime=SingleThreadedAgentRuntime(),
    )


if __name__ == "__main__":
    asyncio.run(main())
