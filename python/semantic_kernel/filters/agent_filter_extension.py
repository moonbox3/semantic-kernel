# Copyright (c) Microsoft. All rights reserved.

from abc import ABC
from collections.abc import Awaitable, Callable, Coroutine
from functools import partial
from typing import Any, Literal, TypeVar

from pydantic import Field

from semantic_kernel.exceptions.filter_exceptions import FilterManagementException
from semantic_kernel.filters.filter_context_base import FilterContextBase
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.kernel_pydantic import KernelBaseModel

_CONTEXT_T = TypeVar("_CONTEXT_T", bound=FilterContextBase)
CALLABLE_FILTER = Callable[
    [_CONTEXT_T, Callable[[_CONTEXT_T], Awaitable[None]]],
    Awaitable[None],
]

ALLOWED = Literal[
    FilterTypes.PROMPT_RENDERING,
    FilterTypes.FUNCTION_INVOCATION,
    FilterTypes.AUTO_FUNCTION_INVOCATION,
]

_MAPPING = {
    FilterTypes.PROMPT_RENDERING: "prompt_rendering_filters",
    FilterTypes.FUNCTION_INVOCATION: "function_invocation_filters",
    FilterTypes.AUTO_FUNCTION_INVOCATION: "auto_function_invocation_filters",
}


class AgentFilterExtension(KernelBaseModel, ABC):
    """Attach, remove, and execute filters at the agent level."""

    prompt_rendering_filters: list[tuple[int, CALLABLE_FILTER]] = Field(default_factory=list)
    function_invocation_filters: list[tuple[int, CALLABLE_FILTER]] = Field(default_factory=list)
    auto_function_invocation_filters: list[tuple[int, CALLABLE_FILTER]] = Field(default_factory=list)

    def add_filter(self, filter_type: ALLOWED | FilterTypes, fn: CALLABLE_FILTER) -> None:
        """Add a filter to the agent."""
        try:
            if not isinstance(filter_type, FilterTypes):
                filter_type = FilterTypes(filter_type)
            getattr(self, _MAPPING[filter_type]).insert(0, (id(fn), fn))
        except Exception as exc:
            raise FilterManagementException(f"Cannot add filter {fn} for {filter_type}") from exc

    def filter(self, filter_type: ALLOWED | FilterTypes):
        """Decorator to register a filter function."""

        def decorator(fn: CALLABLE_FILTER) -> CALLABLE_FILTER:
            self.add_filter(filter_type, fn)
            return fn

        return decorator

    def remove_filter(
        self,
        *,
        filter_type: ALLOWED | FilterTypes | None = None,
        filter_id: int | None = None,
        position: int | None = None,
    ) -> None:
        """Remove a filter from the agent by id or position."""
        if filter_type and not isinstance(filter_type, FilterTypes):
            filter_type = FilterTypes(filter_type)

        if filter_id is None and position is None:
            raise FilterManagementException("Either filter_id or position must be supplied")

        # Remove by position
        if position is not None:
            if filter_type is None:
                raise FilterManagementException("Position removal requires filter_type")
            getattr(self, _MAPPING[filter_type]).pop(position)
            return

        # Remove by id
        targets = (_MAPPING[filter_type],) if filter_type else _MAPPING.values()
        for lst_name in targets:
            lst = getattr(self, lst_name)
            for idx, (fid, _) in enumerate(lst):
                if fid == filter_id:
                    lst.pop(idx)
                    return

    def construct_call_stack(
        self,
        *,
        filter_type: FilterTypes,
        inner_function: Callable[[_CONTEXT_T], Coroutine[Any, Any, None]],
    ) -> Callable[[_CONTEXT_T], Coroutine[Any, Any, None]]:
        """Construct a call stack for the given filter type and inner function."""
        call_chain: list[Any] = [inner_function]
        for _, flt in getattr(self, _MAPPING[filter_type]):
            call_chain.insert(0, partial(flt, next=call_chain[0]))
        return call_chain[0]


def _rebuild_auto_function_invocation_context() -> None:
    from semantic_kernel.agents.agent import Agent  # noqa: F401
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings  # noqa: F401
    from semantic_kernel.contents import FunctionCallContent  # noqa: F401
    from semantic_kernel.contents.chat_history import ChatHistory  # noqa: F401
    from semantic_kernel.filters.agent_contexts import (
        AutoFunctionInvocationContext,
    )
    from semantic_kernel.functions.kernel_arguments import KernelArguments  # noqa: F401
    from semantic_kernel.functions.kernel_function import KernelFunction  # noqa: F401

    AutoFunctionInvocationContext.model_rebuild()


def _rebuild_prompt_render_context() -> None:
    from semantic_kernel.agents.agent import Agent  # noqa: F401
    from semantic_kernel.filters.agent_contexts import PromptRenderContext
    from semantic_kernel.functions.kernel_arguments import KernelArguments  # noqa: F401
    from semantic_kernel.functions.kernel_function import KernelFunction  # noqa: F401

    PromptRenderContext.model_rebuild()


def _rebuild_function_invocation_context() -> None:
    from semantic_kernel.agents.agent import Agent  # noqa: F401
    from semantic_kernel.filters.agent_contexts import FunctionInvocationContext
    from semantic_kernel.functions.kernel_arguments import KernelArguments  # noqa: F401
    from semantic_kernel.functions.kernel_function import KernelFunction  # noqa: F401

    FunctionInvocationContext.model_rebuild()
