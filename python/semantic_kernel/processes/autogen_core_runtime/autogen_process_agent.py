# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import logging
from queue import Queue
from typing import Any

from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime

from semantic_kernel.exceptions.process_exceptions import ProcessEventUndefinedException
from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.autogen_core_runtime.autogen_message_factory import AGMessageFactory
from semantic_kernel.processes.autogen_core_runtime.autogen_process_message import AGProcessMessage
from semantic_kernel.processes.autogen_core_runtime.autogen_step_agent import AGStepAgent
from semantic_kernel.processes.const import END_PROCESS_ID
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_info import KernelProcessStepInfo
from semantic_kernel.processes.local_runtime.local_event import LocalEvent
from semantic_kernel.processes.local_runtime.local_message import LocalMessage
from semantic_kernel.processes.process_message import ProcessMessage

logger: logging.Logger = logging.getLogger(__name__)


class AGProcessAgent(AGStepAgent):
    """A high-level "process" agent that orchestrates sub-step agents.

    in AutoGen's SingleThreadedAgentRuntime. Includes an internal queue
    (`_internal_queue`) and an `internal_execute` method to poll for
    events/messages, then route them via `send_message`.
    """

    def __init__(
        self,
        description: str,
        kernel: Kernel,
        process: KernelProcess,
        runtime: SingleThreadedAgentRuntime,
        parent_process_id: str | None = None,
    ):
        """Create the AGProcessAgent."""
        super().__init__(description, kernel, process, parent_process_id=parent_process_id)
        self.kernel = kernel
        self.process = process
        self.ag_runtime = runtime
        self.parent_process_id = parent_process_id
        self.external_event_queue: Queue = Queue()

        # Keep a local copy of edges from the process so we can do
        # "route messages" to next steps
        self.output_edges = process.output_edges

        # All sub-step agents
        self.steps: list[AgentId] = []

        # Internal queue for events we receive (KernelProcessEvent, ProcessMessage, etc.)
        self._internal_queue: Queue[Any] = Queue()

        # Flag to manage a background "internal_execute" task
        self._execute_task: asyncio.Task | None = None

        self.process_task: asyncio.Task | None = None
        self.initialize_task: bool | None = False

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        """Handle messages addressed to this 'process' agent.

        Instead of routing them immediately, we enqueue them in `_internal_queue`.
        """
        match message:
            case KernelProcessEvent():
                logger.info(f"[AGProcessAgent] Received KernelProcessEvent: {message.id}")
                self._internal_queue.put_nowait(message)

            case ProcessMessage():
                logger.info(f"[AGProcessAgent] Received a ProcessMessage: {message}")
                self._internal_queue.put_nowait(message)

            case AGProcessMessage():
                logger.info(f"[AGProcessAgent] Received an AGProcessMessage: {message}")
                self._internal_queue.put_nowait(message)

            case _:
                logger.info(f"[AGProcessAgent] Received unknown message type: {type(message)}")
                self._internal_queue.put_nowait(message)

        return None

    async def initialize_process(self) -> None:
        """Called right after constructing the agent. We register sub-step agents."""
        await self._init_sub_steps()

    async def _init_sub_steps(self):
        """Create an AGStepAgent for each step in self.process.steps."""
        # Initialize the input and output edges for the process
        self.output_edges = {kvp[0]: list(kvp[1]) for kvp in self.process.edges.items()}

        for step_info in self.process.steps:

            def step_factory(step_info=step_info):
                return AGStepAgent(
                    f"StepAgent_{step_info.state.name}_{step_info.state.id}",
                    self.kernel,
                    step_info,
                    parent_process_id=self.id.key if isinstance(self.id, AgentId) else None,
                )

            agent_id = AgentId(f"{step_info.state.id}", step_info.state.id)

            logger.info(
                f"[AGProcessAgent] Registering sub-step agent: {agent_id} with step type: {step_info.inner_step_type}"
            )

            await self.ag_runtime.register_factory(
                agent_id.type,
                agent_factory=step_factory,
                expected_class=AGStepAgent,
            )
            self.steps.append(agent_id)

    async def initialize_step(self):
        """Initializes the step."""
        # The process does not need any further initialization
        pass

    async def ensure_initialized(self):
        """Ensures the process is initialized lazily (synchronously)."""
        if not self.initialize_task:
            await self.initialize_process()
            self.initialize_task = True

    async def start(self, keep_alive: bool = True):
        """Starts the process with an initial event."""
        await self.ensure_initialized()
        self.process_task = asyncio.create_task(self.internal_execute(keep_alive=keep_alive))
        self.ag_runtime.start()

    async def run_once(self, process_event: KernelProcessEvent):
        """Starts the process with an initial event and waits for it to finish."""
        if process_event is None:
            raise ProcessEventUndefinedException("The process event must be specified.")
        self.external_event_queue.put(process_event)
        await self.start(keep_alive=False)
        if self.process_task:
            await self.process_task

    async def internal_execute(self, max_supersteps: int = 100, keep_alive: bool = True):
        """Main loop for processing items from the queue until stop condition.

        This is analogous to local_runtime.local_process.internal_execute.
        """
        message_channel: Queue[LocalMessage] = Queue()
        try:
            for _ in range(max_supersteps):
                self.enqueue_external_messages(message_channel)
                for step in self.steps:
                    await self.enqueue_step_messages(step, message_channel)

                messages_to_process: list[LocalMessage] = []
                while not message_channel.empty():
                    messages_to_process.append(message_channel.get())

                if not messages_to_process and (not keep_alive or self.external_event_queue.empty()):
                    break

                message_tasks = []
                for message in messages_to_process:
                    if message.destination_id == END_PROCESS_ID:
                        break

                    destination_step = next(step for step in self.steps if step.key == message.destination_id)
                    message_tasks.append(self.send_message(message, recipient=destination_step))

                await asyncio.gather(*message_tasks)

        except asyncio.CancelledError:
            logger.info("[AGProcessAgent] internal_execute got cancelled.")
        except Exception as e:
            logger.error(f"[AGProcessAgent] internal_execute encountered error: {e}", exc_info=True)

    async def get_process_info(self) -> KernelProcess:
        """Return an updated KernelProcess from each sub-step agent.

        If steps have changed state, we can gather it back here.
        """
        new_steps: list[KernelProcessStepInfo] = []
        for step_agent in self.steps:
            agent = await self.ag_runtime.try_get_underlying_agent_instance(step_agent, type=AGStepAgent)
            new_steps.append(agent.step_info)
        return KernelProcess(
            state=self.process.state,
            steps=new_steps,
            edges=self.output_edges,
        )

    async def dispose(self) -> None:
        """Cleanup resources if needed."""
        if self._execute_task and not self._execute_task.done():
            self._execute_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._execute_task

    def enqueue_external_messages(self, message_channel: Queue[LocalMessage]):
        """Processes external events that have been sent to the process."""
        while not self.external_event_queue.empty():
            external_event: KernelProcessEvent = self.external_event_queue.get_nowait()
            if external_event.id in self.output_edges:
                edges = self.output_edges[external_event.id]
                for edge in edges:
                    message = AGMessageFactory.create_from_edge(edge, external_event.data)
                    message_channel.put(message)

    async def enqueue_step_messages(self, step: AGStepAgent, message_channel: Queue[LocalMessage]):
        """Processes events emitted by the given step and enqueues them."""
        agent_step = await self.ag_runtime.try_get_underlying_agent_instance(
            AgentId(step.type, step.key), type=AGStepAgent
        )
        all_step_events = agent_step.get_all_events()
        for step_event in all_step_events:
            if step_event.visibility == KernelProcessEventVisibility.Public:
                if isinstance(step_event, KernelProcessEvent):
                    await self.emit_event(step_event)  # type: ignore
                elif isinstance(step_event, LocalEvent):
                    await self.emit_local_event(step_event)  # type: ignore

            for edge in agent_step.get_edge_for_event(step_event.id):
                message = AGMessageFactory.create_from_edge(edge, step_event.data)
                message_channel.put(message)
