# Copyright (c) Microsoft. All rights reserved.


from pydantic import Field

from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.utils.experimental_decorator import experimental_class


@experimental_class
class OpenAIAssistantExecutionOptions(KernelBaseModel):
    """OpenAI Assistant Execution Settings class."""

    max_completion_tokens: int | None = Field(None)
    max_prompt_tokens: int | None = Field(None)
    parallel_tool_calls_enabled: bool | None = Field(True)
    truncation_message_count: int | None = Field(None)