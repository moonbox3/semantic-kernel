# Copyright (c) Microsoft. All rights reserved.

from enum import Enum
from typing import TYPE_CHECKING, Any

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.exceptions.process_exceptions import ProcessInvalidConfigurationException
from semantic_kernel.processes.autogen_runtime.autogen_kernel_process_context import AutoGenKernelProcessContext
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

if TYPE_CHECKING:
    from semantic_kernel.processes.kernel_process import KernelProcess


async def start(
    runtime: SingleThreadedAgentRuntime,
    process: "KernelProcess",
    initial_event: KernelProcessEvent | str | Enum,
    process_id: str | None = None,
    **kwargs: Any,
) -> AutoGenKernelProcessContext:
    """Start the kernel process with SingleThreadedAgentRuntime."""
    if process is None:
        raise ProcessInvalidConfigurationException("process cannot be None")
    if process.state is None:
        raise ProcessInvalidConfigurationException("process state cannot be empty")
    if initial_event is None:
        raise ProcessInvalidConfigurationException("initial_event cannot be None")

    if isinstance(initial_event, Enum):
        initial_event = initial_event.value

    if isinstance(initial_event, str):
        initial_event = KernelProcessEvent(id=initial_event, data=kwargs.get("data"))

    if process_id is not None:
        process.state.id = process_id

    # Create an AutoGen kernel process context
    process_context = AutoGenKernelProcessContext(process, runtime)
    await process_context.start_with_event(initial_event)
    return process_context
