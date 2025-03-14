# Copyright (c) Microsoft. All rights reserved.

from enum import Enum
from typing import TYPE_CHECKING, Any

from semantic_kernel.exceptions.process_exceptions import ProcessInvalidConfigurationException
from semantic_kernel.processes.core_runtime.core_kernel_process_context import CoreKernelProcessContext
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

if TYPE_CHECKING:
    from semantic_kernel.processes.kernel_process import KernelProcess


async def start(
    process: "KernelProcess",
    initial_event: KernelProcessEvent | str | Enum,
    process_id: str | None = None,
    **kwargs: Any,
) -> CoreKernelProcessContext:
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

    if process_id is not None and process.state.id is None:
        process.state.id = process_id

    # Create a Core kernel process context
    process_context = CoreKernelProcessContext(process)
    await process_context.start_with_event(initial_event)
    return process_context
