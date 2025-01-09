# Copyright (c) Microsoft. All rights reserved.

import logging
import uuid
from typing import Any

from autogen_core import AgentId, BaseAgent, MessageContext, SingleThreadedAgentRuntime

from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.autogen_core_runtime.autogen_message_factory import AGMessageFactory
from semantic_kernel.processes.autogen_core_runtime.autogen_step_agent import AGStepAgent
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_info import (
    KernelProcessStepInfo,
)
from semantic_kernel.processes.process_message import ProcessMessage

logger: logging.Logger = logging.getLogger(__name__)


class AGProcessAgent(BaseAgent):
    """A high-level "process" agent that orchestrates sub-step agents in AutoGen's SingleThreadedAgentRuntime.

    Instead of manually looping over events (like
    local_runtime), this agent can:
      1. Register sub-step agents,
      2. Convert an initial `KernelProcessEvent` into an AutoGen message and send it,
      3. Potentially route further events by calling 'some_method_when_ready_to_emit'
         which sends messages to sub-steps for relevant edges.

    If have external triggers, we can do `await self.send_message(...)`
    from outside, or direct them into `on_message_impl(...)`.
    """

    def __init__(
        self,
        description: str,
        kernel: Kernel,
        process: KernelProcess,
        runtime: SingleThreadedAgentRuntime,
        parent_process_id: str | None = None,
        proc_id: str | None = None,
    ):
        """Create the AGProcessAgent."""
        super().__init__(description)
        self.kernel = kernel
        self.process = process
        self.ag_runtime = runtime
        self.parent_process_id = parent_process_id

        # We'll keep a local copy of output_edges from the process so we can do
        # "ready_to_emit(event_id, data)" to route to next steps
        self.output_edges = process.output_edges

        # We'll store references to the sub-step agents we create
        self.steps: list[AGStepAgent] = []
        # Unique for an AGProcess, if we want to create multiple processes
        self.id_str: str = process.state.id or uuid.uuid4().hex  # not used?

        self.proc_id = proc_id

    @property
    def ag_type(self) -> str:
        """Must be unique among concurrently-registered agents in SingleThreadedAgentRuntime.

        Use f'ag_process_{self.proc_id}' for uniqueness.
        """
        return self.proc_id

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        """Handle any messages addressed to this 'process' agent.

        For instance, we might get a KernelProcessEvent directly from outside, or a
        ProcessMessage from a sub-step, etc.
        """
        match message:
            case KernelProcessEvent():
                logger.info(f"[AGProcessAgent] Received KernelProcessEvent: {message.id}")
                await self.ready_to_emit(message.id, message.data)

            case ProcessMessage():
                logger.info(f"[AGProcessAgent] Received a ProcessMessage: {message}")
                # handle or ignore

            case _:
                logger.info(f"[AGProcessAgent] Received unknown message type: {type(message)}")

        return None

    async def initialize(self) -> None:
        """Called after constructing the agent. We do a sub-step registration here."""
        await self._init_sub_steps()

    async def _init_sub_steps(self):
        """For each step in self.process.steps, create an AGStepAgent and register it.

        Each step's 'type' must be unique in the runtime.
        """
        for step_info in self.process.steps:
            step_description = f"StepAgent_{step_info.state.name}_{step_info.state.id}"
            step_agent = AGStepAgent(step_description, self.kernel, step_info, parent_process_id=self.id.key)

            # Each step agent must have a unique "type" for the SingleThreadedAgentRuntime registry:
            agent_type_str = f"ag_step_{step_info.state.id or uuid.uuid4().hex}"

            agent_id = AgentId(agent_type_str, step_info.state.id or uuid.uuid4().hex)

            def step_factory():
                return step_agent

            await self.ag_runtime.register_factory(
                agent_id.type,
                agent_factory=step_factory,
                expected_class=AGStepAgent,
            )
            self.steps.append(step_agent)

    async def run_once(self, initial_event: KernelProcessEvent) -> None:
        """Run the process once by kicking off the initial event.

        We send the KernelProcessEvent itself to this process
        so that 'on_message_impl' is triggered. Then let single-threaded runtime do the rest.
        """
        # We can just send the event directly to self
        await self.send_message(initial_event, recipient=self.ag_type)

        # If we want to wait until the system is idle...
        await self.ag_runtime.stop_when_idle()

    async def ready_to_emit(self, event_id: str, data: Any) -> None:
        """Example function that replicates 'enqueue_step_messages' from local runtime.

        For a given event_id, find edges, build an AGMessage, and send it to each next step.
        """
        matching_edges = self.output_edges.get(event_id, [])
        for edge in matching_edges:
            # Convert edge + data => AGProcessMessage or similar
            msg = AGMessageFactory.create_from_edge(edge, data)

            # build the agent's type the same way we did in `_init_sub_steps` for that step
            step_type_str = f"ag_step_{edge.output_target.step_id}"
            # the key is the step id
            step_key = edge.output_target.step_id

            next_step_id = AgentId(step_type_str, step_key)

            # Now actually send the message to that step
            await self.send_message(msg, recipient=next_step_id)

    async def get_process_info(self) -> KernelProcess:
        """Return an updated KernelProcess object from each sub-step agent.

        For example, if each sub-step agent changed state, we can gather it.
        """
        new_steps: list[KernelProcessStepInfo] = []
        for step_agent in self.steps:
            # Or we might call step_agent.some_method to gather updated step_info, etc.
            new_steps.append(step_agent.step_info)
        # Return a new KernelProcess with the updated steps
        return KernelProcess(state=self.process.state, steps=new_steps, edges=self.output_edges)

    async def dispose(self) -> None:
        """Clean up resources, if needed."""
        pass
