# Copyright (c) Microsoft. All rights reserved.

import uuid

from autogen_core import AgentId, SingleThreadedAgentRuntime

from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.core_runtime.core_process import CoreProcess
from semantic_kernel.processes.core_runtime.core_process_info import CoreProcessInfo
from semantic_kernel.processes.core_runtime.core_step import CoreStep
from semantic_kernel.processes.core_runtime.event_buffer_agent import EventBufferAgent
from semantic_kernel.processes.core_runtime.external_event_buffer_agent import ExternalEventBufferAgent
from semantic_kernel.processes.core_runtime.message_buffer_agent import MessageBufferAgent
from semantic_kernel.processes.core_runtime.messages import (
    GetProcessInfoMessage,
    InitializeStepMessage,
    RunOnceMessage,
    SendProcessMessage,
    StopProcessMessage,
)
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent


class CoreKernelProcessContext:
    """The Core Kernel Process Context."""

    def __init__(self, process: KernelProcess):
        """Initialize a new instance of CoreKernelProcessContext."""
        if not process.state.name:
            raise ValueError("Process state name must not be empty")
        if not process.state.id:
            process.state.id = str(uuid.uuid4().hex)

        self.process = process
        self.process_agent_id = AgentId("CoreProcess", process.state.id)
        self.runtime = SingleThreadedAgentRuntime()

    async def start_with_event(self, initial_event: KernelProcessEvent) -> None:
        """Initialize the process with a message, then run once with the event."""
        # Build the process info CoreProcessInfo from the KernelProcess

        await self.register_core_components()

        process_info = CoreProcessInfo.from_kernel_process(self.process)

        init_msg = InitializeStepMessage(
            step_info_str=process_info.model_dump_json(),
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

        process_info = CoreProcessInfo.model_validate(info)
        return process_info.to_kernel_process()

    async def register_core_components(
        self,
    ) -> None:
        """Registers agent types with SingleThreadedAgentRuntime."""
        # if factories is None:
        factories = {}

        async def process_agent_factory():  # noqa: RUF029
            return CoreProcess(self.process_agent_id, Kernel(), factories, self.runtime)

        await self.runtime.register_factory("CoreProcess", agent_factory=process_agent_factory)

        async def step_agent_factory():  # noqa: RUF029
            return CoreStep(self.process_agent_id, Kernel(), factories, self.runtime)

        await self.runtime.register_factory("CoreStep", agent_factory=step_agent_factory)

        async def message_buffer_factory():  # noqa: RUF029
            return MessageBufferAgent(AgentId("MessageBufferAgent", "MessageBufferAgent"))

        await self.runtime.register_factory(
            "MessageBufferAgent",
            agent_factory=message_buffer_factory,
        )

        async def event_buffer_factory():  # noqa: RUF029
            return EventBufferAgent(AgentId("EventBufferAgent", "EventBufferAgent"))

        await self.runtime.register_factory(
            "EventBufferAgent",
            agent_factory=event_buffer_factory,
        )

        async def external_event_buffer_factory():  # noqa: RUF029
            return ExternalEventBufferAgent(AgentId("ExternalEventBufferAgent", "ExternalEventBufferAgent"))

        await self.runtime.register_factory(
            "ExternalEventBufferAgent",
            agent_factory=external_event_buffer_factory,
        )

        try:
            self.runtime.start()
        except Exception as e:
            print(f"Error starting runtime: {e}")
            raise
