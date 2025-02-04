# Copyright (c) Microsoft. All rights reserved.

from enum import Enum
from typing import TYPE_CHECKING

from semantic_kernel.exceptions.process_exceptions import ProcessInvalidConfigurationException
from semantic_kernel.processes.autogen_core_runtime.autogen_kernel_process_context import AGKernelProcessContext
from semantic_kernel.processes.local_runtime.local_event import KernelProcessEvent
from semantic_kernel.utils.experimental_decorator import experimental_function

if TYPE_CHECKING:
    from semantic_kernel.kernel import Kernel
    from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess


@experimental_function
async def start(
    process: "KernelProcess", kernel: "Kernel", initial_event: KernelProcessEvent | str | Enum, **kwargs
) -> AGKernelProcessContext:
    """Start the kernel process using the autogen core runtime."""
    if process is None:
        raise ProcessInvalidConfigurationException("process cannot be None")
    if process.state is None or not process.state.name:
        raise ProcessInvalidConfigurationException("process state name cannot be empty")
    if kernel is None:
        raise ProcessInvalidConfigurationException("kernel cannot be None")
    if initial_event is None:
        raise ProcessInvalidConfigurationException("initial_event cannot be None")

    initial_event_obj = (
        KernelProcessEvent(id=initial_event.value, data=kwargs.get("data"))
        if isinstance(initial_event, Enum)
        else (
            KernelProcessEvent(id=initial_event, data=kwargs.get("data"))
            if isinstance(initial_event, str)
            else initial_event
        )
    )

    process_context = AGKernelProcessContext(process, kernel)
    await process_context.start_with_event(initial_event_obj)
    return process_context
