# Copyright (c) Microsoft. All rights reserved.
import logging
import uuid

from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime

from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.autogen_core_runtime.autogen_cancellation_token import AGCancellationToken
from semantic_kernel.processes.autogen_core_runtime.autogen_process_agent import AGProcessAgent
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

logger: logging.Logger = logging.getLogger(__name__)


class AGKernelProcessContext:
    """This class parallels the local_runtime.local_kernel_process_context approach."""

    def __init__(self, process: KernelProcess, kernel: Kernel):
        """Constructor that saves the process, kernel, and runtime references."""
        if not process.state.name:
            raise ValueError("Process state name must not be empty")

        self.runtime = SingleThreadedAgentRuntime()
        self.kernel = kernel
        self.process = process

        proc_id = process.state.id or uuid.uuid4().hex
        self.proc_agent_id = AgentId(f"{proc_id}", proc_id)

    async def __aenter__(self):
        """Create & register the agent factory here, as an async context init step."""

        def process_factory():
            return AGProcessAgent(
                description=f"ProcessAgent_{self.process.state.name}",
                kernel=self.kernel,
                process=self.process,
                runtime=self.runtime,
                parent_process_id=None,
            )

        logger.info(f"[AGProcessAgent] Registering process agent: {self.proc_agent_id}")

        await self.runtime.register_factory(
            type=self.proc_agent_id.type,
            agent_factory=process_factory,
            expected_class=AGProcessAgent,
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup logic, if needed."""
        await self.dispose()

    async def start_with_event(self, initial_event: KernelProcessEvent):
        """Example of how to retrieve the actual agent instance and call `run_once`."""
        agent_instance = await self.runtime.try_get_underlying_agent_instance(self.proc_agent_id, type=AGProcessAgent)
        await agent_instance.run_once(initial_event)

    async def send_event(self, event: KernelProcessEvent):
        """Send an event to the process agent."""
        agent_instance = await self.runtime.try_get_underlying_agent_instance(self.proc_agent_id, type=AGProcessAgent)
        await agent_instance.on_message_impl(
            event,
            MessageContext(
                sender=None,
                topic_id=None,
                is_rpc=False,
                cancellation_token=AGCancellationToken(),
                message_id=None,
            ),
        )

    async def stop(self):
        """Stop the process agent."""
        agent_instance = await self.runtime.try_get_underlying_agent_instance(self.proc_agent_id, type=AGProcessAgent)
        await agent_instance.stop()

    async def get_state(self) -> KernelProcess:
        """Get the updated KernelProcess object from the agent."""
        agent_instance = await self.runtime.try_get_underlying_agent_instance(self.proc_agent_id, type=AGProcessAgent)
        return await agent_instance.get_process_info()

    async def dispose(self):
        """Remove agent from runtime or do final cleanup (if needed)."""
        pass
