# Copyright (c) Microsoft. All rights reserved.


from typing import TYPE_CHECKING, Any

from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.filters.filter_context_base import FilterContextBase
from semantic_kernel.functions.kernel_arguments import KernelArguments

if TYPE_CHECKING:
    from semantic_kernel.agents.agent import Agent


class PromptRenderContext(FilterContextBase):
    """Context for prompt rendering filters."""

    agent: "Agent"
    rendered_prompt: str | None = None
    arguments: KernelArguments
    is_streaming: bool = False
    function_result: Any | None = None


class FunctionInvocationContext(FilterContextBase):
    """Context for function invocation filters."""

    agent: "Agent"
    function: Any
    arguments: KernelArguments
    is_streaming: bool = False
    result: Any | None = None


class AutoFunctionInvocationContext(FilterContextBase):
    """Context for auto function invocation filters."""

    agent: "Agent"
    function: Any
    arguments: KernelArguments
    chat_history: ChatHistory
    function_call: FunctionCallContent
    request_sequence_index: int
    function_sequence_index: int
    function_result: Any | None = None
    terminate: bool = False
