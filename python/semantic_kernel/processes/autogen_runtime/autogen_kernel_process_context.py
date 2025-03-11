# Copyright (c) Microsoft. All rights reserved.

import json
import uuid

from autogen_core import AgentId, SingleThreadedAgentRuntime

from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo
from semantic_kernel.processes.autogen_runtime.messages import (
    GetProcessInfoMessage,
    InitializeProcessMessage,
    RunOnceMessage,
    SendProcessMessage,
    StopProcessMessage,
)
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent


class AutoGenKernelProcessContext:
    """A context that manages a single "process" instance in SingleThreadedAgentRuntime.

    References a 'ProcessAgent' that has no decorators.
    """

    def __init__(self, process: KernelProcess, runtime: SingleThreadedAgentRuntime):
        """Initialize a new instance of AutoGenKernelProcessContext."""
        if not process.state.name:
            raise ValueError("Process state name must not be empty")
        if not process.state.id:
            process.state.id = str(uuid.uuid4().hex)

        self.process = process
        self.runtime = runtime
        self.process_agent_id = AgentId("process_agent", process.state.id)

    async def start_with_event(self, initial_event: KernelProcessEvent) -> None:
        """Initialize the process with a message, then run once with the event."""
        # Build the process info AutoGenProcessInfo from the KernelProcess
        from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo

        process_info = AutoGenProcessInfo.from_kernel_process(self.process)

        init_msg = InitializeProcessMessage(
            process_info=process_info,
            parent_process_id=None,
        )
        await self.runtime.send_message(init_msg, self.process_agent_id)

        run_once_msg = RunOnceMessage(process_event=initial_event)
        await self.runtime.send_message(run_once_msg, self.process_agent_id)

    async def send_event(self, event: KernelProcessEvent) -> None:
        """Queue a process event."""
        msg = SendProcessMessage(process_event=event)
        await self.runtime.send_message(msg, self.process_agent_id)

    async def stop(self) -> None:
        """Stop the process."""
        stop_msg = StopProcessMessage()
        await self.runtime.send_message(stop_msg, self.process_agent_id)

    async def get_state(self) -> KernelProcess:
        """Retrieve current state by sending a GetProcessInfoMessage, then reconstructing the KernelProcess."""
        info = await self.runtime.send_message(GetProcessInfoMessage(), self.process_agent_id)

        autogen_proc_info = AutoGenProcessInfo.model_validate(json.loads(info))
        return autogen_proc_info.to_kernel_process()
