# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import MutableSequence
from queue import Queue
from typing import TYPE_CHECKING, Any

from agent_runtime import AgentId, CoreAgentId, InProcessRuntime, MessageContext

from semantic_kernel.exceptions.process_exceptions import ProcessEventUndefinedException
from semantic_kernel.processes.const import END_PROCESS_ID
from semantic_kernel.processes.core_runtime.core_process_info import CoreProcessInfo
from semantic_kernel.processes.core_runtime.core_step import CoreStep
from semantic_kernel.processes.core_runtime.core_step_info import CoreStepInfo
from semantic_kernel.processes.core_runtime.messages import (
    DequeueAllMessages,
    EnqueueExternalEvent,
    EnqueueMessage,
    GetProcessInfoMessage,
    InitializeStepMessage,
    PrepareIncomingMessagesMessage,
    ProcessIncomingMessagesMessage,
    RunOnceMessage,
    SendProcessMessage,
    StartProcessMessage,
    StopProcessMessage,
    ToCoreStepInfoMessage,
)
from semantic_kernel.processes.kernel_process.kernel_process_edge import KernelProcessEdge
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)
from semantic_kernel.processes.kernel_process.kernel_process_state import KernelProcessState
from semantic_kernel.processes.process_event import ProcessEvent
from semantic_kernel.processes.process_message_factory import ProcessMessageFactory

if TYPE_CHECKING:
    from semantic_kernel.kernel import Kernel

logger = logging.getLogger(__name__)


class CoreProcess(CoreStep):
    """A "Process" agent that inherits from StepAgent."""

    def __init__(self, agent_id: AgentId, kernel: "Kernel", factories: dict[str, Any], runtime: InProcessRuntime):
        """Initialize the ProcessAgent."""
        super().__init__(agent_id, kernel, factories, runtime)
        self._id = agent_id
        self._runtime = runtime
        self.kernel = kernel

        self.process: CoreProcessInfo | None = None
        self.steps: list[CoreStepInfo] = []
        self.step_infos: MutableSequence[CoreStepInfo] = []
        self.external_event_queue: Queue[str] = Queue()
        self.process_task: asyncio.Task | None = None

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """We'll do an `if isinstance(...)` check for *process-level* messages first.

        Otherwise, we fallback to StepAgent's version.
        """
        if isinstance(message, InitializeStepMessage):
            return await self._handle_initialize_process(message)
        if isinstance(message, StartProcessMessage):
            return await self._handle_start_process(message)
        if isinstance(message, RunOnceMessage):
            return await self._handle_run_once(message)
        if isinstance(message, StopProcessMessage):
            return await self._handle_stop_process(message)
        if isinstance(message, SendProcessMessage):
            return await self._handle_send_process_message(message)
        if isinstance(message, GetProcessInfoMessage):
            return await self._handle_get_process_info(message)

        raise ValueError(f"Unknown message type: {type(message)}")

    async def _handle_initialize_process(self, msg: InitializeStepMessage):
        """Called to set up this process as if it were a step.

        But we also create sub-steps or nested processes for each child in process_info.
        """
        if self.initialize_task:
            return

        core_process_info = CoreProcessInfo.model_validate(json.loads(msg.step_info_str))

        self.parent_process_id = msg.parent_process_id
        self.process = core_process_info
        self.step_infos = list(self.process.steps)
        self.output_edges = {kvp[0]: list(kvp[1]) for kvp in self.process.edges.items()}

        # For each sub-step, we create a StepAgent or another ProcessAgent
        for step in self.step_infos:
            # The current step should already have a name.
            assert step.state and step.state.name is not None  # nosec

            if isinstance(step, CoreProcessInfo):
                # The process will only have an Id if it's already been executed.
                if not step.state.id:
                    step.state.id = str(uuid.uuid4().hex)
                # nested process
                scoped_process_id = self._scoped_actor_id(CoreAgentId("CoreProcess", step.state.id))
                init_msg = InitializeStepMessage(step_info_str=step.model_dump_json(), parent_process_id=self._id.key)
                await self._runtime.send_message(init_msg, scoped_process_id)
            else:
                # The current step should already have an Id.
                assert step.state and step.state.id is not None  # nosec
                scoped_step_id = self._scoped_actor_id(CoreAgentId("CoreStep", step.state.id))
                init_msg = InitializeStepMessage(step_info_str=step.model_dump_json(), parent_process_id=self._id.key)
                await self._runtime.send_message(init_msg, scoped_step_id)

            self.steps.append(step)

        logger.info(f"[ProcessAgent {self._id.key}] initialized with {len(self.steps)} steps.")
        self.initialize_task = True

    async def _handle_start_process(self, msg: StartProcessMessage):
        if not self.initialize_task:
            raise ValueError("Process not initialized")

        if not self.process_task or self.process_task.done():
            self.process_task = asyncio.create_task(self._internal_execute(keep_alive=msg.keep_alive))

    async def _handle_run_once(self, msg: RunOnceMessage):
        if not msg.process_event:
            raise ProcessEventUndefinedException("Must supply a process_event.")
        # Enqueue the event in external buffer
        ext_event_buf_id = CoreAgentId("ExternalEventBufferAgent", self._id.key)
        await self._runtime.send_message(
            EnqueueMessage(message_json=msg.process_event.model_dump_json()), ext_event_buf_id
        )

        # then start with keep_alive=False
        await self._handle_start_process(StartProcessMessage(keep_alive=False))
        if self.process_task:
            try:
                await self.process_task
            except asyncio.CancelledError:
                logger.error("[ProcessAgent] process_task was cancelled")

    async def _handle_stop_process(self, msg: StopProcessMessage):
        if not self.process_task or self.process_task.done():
            return
        self.process_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.process_task

    async def _handle_send_process_message(self, msg: SendProcessMessage):
        if not msg.process_event:
            raise ValueError("No event specified")
        # Enqueue in external buffer
        ext_buf_id = CoreAgentId("ExternalEventBufferAgent", self._id.key)
        await self._runtime.send_message(
            EnqueueExternalEvent(event_json=msg.process_event.model_dump_json()), ext_buf_id
        )

    async def _handle_get_process_info(self, msg: GetProcessInfoMessage) -> dict:
        if self.process is None:
            raise ValueError("The process must be initialized before converting to CoreProcessInfo.")
        if self.process.inner_step_python_type is None:
            raise ValueError("The inner step type must be defined before converting to CoreProcessInfo.")

        process_state = KernelProcessState(name=self.name, id=self._id.key)

        step_tasks = [self._get_step_as_process_info(step) for step in self.steps]
        steps = await asyncio.gather(*step_tasks)

        core_process_info = CoreProcessInfo(
            inner_step_python_type=self.process.inner_step_python_type,
            edges=self.process.edges,
            state=process_state,
            steps=steps,
        )

        return core_process_info.model_dump()

    async def _get_step_as_process_info(self, step: CoreStepInfo | CoreProcessInfo) -> Any:
        """Get a step as process info."""
        agent_type = self._get_agent_type(step)
        scoped_step_id = self._scoped_actor_id(CoreAgentId(agent_type, step.state.id))
        return await self._runtime.send_message(ToCoreStepInfoMessage(), scoped_step_id)

    async def _internal_execute(self, max_supersteps=100, keep_alive=True):
        """The main loop for delivering external events, step messages, etc."""
        for _ in range(max_supersteps):
            if await self._check_end():
                break

            # external events
            await self._enqueue_external_events()

            step_prepare_tasks = [self._prepare_incoming_messages(step) for step in self.steps]
            message_counts = await asyncio.gather(*step_prepare_tasks)

            if sum(message_counts) == 0 and (not keep_alive or self.external_event_queue.empty()):
                # Exit the loop without cancelling the task
                break

            step_process_tasks = [self._process_incoming_messages(step) for step in self.steps]
            await asyncio.gather(*step_process_tasks)

            # handle public events
            await self._handle_public_events()

    async def _prepare_incoming_messages(self, step: CoreStepInfo | CoreProcessInfo) -> int:
        agent_type = self._get_agent_type(step)
        scoped_step_id = self._scoped_actor_id(CoreAgentId(agent_type, step.state.id))
        return await self._runtime.send_message(PrepareIncomingMessagesMessage(), scoped_step_id)

    async def _process_incoming_messages(self, step: CoreStepInfo | CoreProcessInfo) -> None:
        agent_type = self._get_agent_type(step)
        scoped_step_id = self._scoped_actor_id(CoreAgentId(agent_type, step.state.id))
        await self._runtime.send_message(ProcessIncomingMessagesMessage(), scoped_step_id)

    def _get_agent_type(self, step: CoreStepInfo | CoreProcessInfo) -> str:
        """Get the agent type based on the step type."""
        if isinstance(step, CoreStepInfo):
            return "CoreStep"
        if isinstance(step, CoreProcessInfo):
            return "CoreProcess"
        raise ValueError(f"Unknown step type: {step.type}")

    async def _check_end(self) -> bool:
        scoped_end_mb_id = self._scoped_actor_id(CoreAgentId("MessageBufferAgent", END_PROCESS_ID))
        leftover = await self._runtime.send_message(DequeueAllMessages(), scoped_end_mb_id)
        return len(leftover) > 0

    async def _enqueue_external_events(self):
        ext_buf_id = CoreAgentId("ExternalEventBufferAgent", self._id.key)
        leftover = await self._runtime.send_message(DequeueAllMessages(), ext_buf_id)

        event_objs = []
        for s in leftover:
            e = json.loads(s)
            ev = KernelProcessEvent.model_validate(e)
            event_objs.append(ev)

        if not self.output_edges:
            return
        for ev in event_objs:
            if ev.id in self.output_edges:
                for edge in self.output_edges[ev.id]:
                    pm = ProcessMessageFactory.create_from_edge(edge, ev.data)
                    target_step = edge.output_target.step_id
                    if target_step:
                        scoped_mb_id = self._scoped_actor_id(CoreAgentId("MessageBufferAgent", target_step))
                        await self._runtime.send_message(
                            EnqueueMessage(message_json=pm.model_dump_json()), scoped_mb_id
                        )

    async def _handle_public_events(self):
        if self.parent_process_id is not None:
            evbuf_id = CoreAgentId("EventBufferAgent", self._id.key)
            leftover = await self._runtime.send_message(DequeueAllMessages(), evbuf_id)

            for s in leftover or []:
                loaded = json.loads(s)
                p_ev = ProcessEvent.model_validate(loaded)
                if (
                    p_ev.inner_event
                    and p_ev.inner_event.visibility == KernelProcessEventVisibility.Public
                    and p_ev.id in self.output_edges
                ):
                    for edge in self.output_edges[p_ev.id]:
                        assert isinstance(edge, KernelProcessEdge)  # nosec
                        pm = ProcessMessageFactory.create_from_edge(edge, p_ev.data)
                        if edge.output_target.step_id:
                            scoped_mb_id = self._scoped_actor_id(
                                CoreAgentId("EventBufferAgent", edge.output_target.step_id), scope_to_parent=True
                            )

                            await self._runtime.send_message(
                                EnqueueMessage(message_json=pm.model_dump_json()), scoped_mb_id
                            )
