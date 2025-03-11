# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import json
import logging
from queue import Queue
from typing import TYPE_CHECKING, Any

from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime

from semantic_kernel.exceptions.process_exceptions import ProcessEventUndefinedException
from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo
from semantic_kernel.processes.autogen_runtime.messages import (
    CountPreparedMessages,
    DequeueAllExternalEvents,
    EnqueueExternalEvent,
    GetProcessInfoMessage,
    InitializeProcessMessage,
    InitializeStepMessage,
    PrepareIncomingMessagesMessage,
    ProcessIncomingMessagesMessage,
    RunOnceMessage,
    SendProcessMessage,
    StartProcessMessage,
    StopProcessMessage,
    ToAutoGenStepInfoMessage,
)
from semantic_kernel.processes.autogen_runtime.step_agent import StepAgent
from semantic_kernel.processes.const import END_PROCESS_ID
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


class ProcessAgent(StepAgent):
    """A "Process" agent that inherits from StepAgent."""

    def __init__(self, agent_id: str, kernel: "Kernel", factories: dict[str, Any], runtime: SingleThreadedAgentRuntime):
        """Initialize the ProcessAgent."""
        super().__init__(agent_id, factories)
        # Now store the runtime
        self._runtime = runtime
        self.kernel = kernel

        self.process: AutoGenProcessInfo | None = None
        self.steps: list[AgentId] = []
        self.external_event_queue: Queue[str] = Queue()
        self.process_task: asyncio.Task | None = None

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """We'll do an `if isinstance(...)` check for *process-level* messages first.

        Otherwise, we fallback to StepAgent's version.
        """
        if isinstance(message, InitializeProcessMessage):
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

        # Not a known process-level message => let the StepAgent handle it
        return await super().on_message_impl(message, context)

    async def _handle_initialize_process(self, msg: InitializeProcessMessage):
        """Called to set up this process as if it were a step.

        But we also create sub-steps or nested processes for each child in process_info.
        """
        if self.initialize_task:
            return

        self.process = msg.process_info
        self.output_edges = dict(self.process.edges)
        self.parent_process_id = msg.parent_process_id

        # We also treat the top-level process as if it were a "step":
        # step_info is just the same data
        self.step_info = self.process
        self.step_state = self.process.state
        self.initialize_task = True

        # For each sub-step, we create a StepAgent or another ProcessAgent
        for step_info in self.process.steps:
            step_id_str = step_info.state.id
            if not step_id_str:
                from uuid import uuid4

                step_id_str = uuid4().hex
                step_info.state.id = step_id_str

            if step_info.type == "AutoGenProcessInfo":
                # nested process
                nested_process_id = AgentId("process_agent", step_id_str)
                init_msg = InitializeProcessMessage(process_info=step_info, parent_process_id=self.id.key)
                await self._runtime.send_message(init_msg, nested_process_id)
                self.steps.append(nested_process_id)
            else:
                # single step
                step_agent_id = AgentId("step_agent", step_id_str)
                step_payload = {"step_info": step_info.model_dump_json(), "parent_process_id": self.id.key}
                init_step_msg = InitializeStepMessage(
                    step_info_json=json.dumps(step_payload), parent_process_id=self.id.key
                )
                await self._runtime.send_message(init_step_msg, step_agent_id)
                self.steps.append(step_agent_id)

        logger.info(f"[ProcessAgent {self.id}] initialized with {len(self.steps)} steps.")

    async def _handle_start_process(self, msg: StartProcessMessage):
        if not self.initialize_task:
            raise ValueError("Process not initialized")

        if not self.process_task or self.process_task.done():
            self.process_task = asyncio.create_task(self._internal_execute(keep_alive=msg.keep_alive))

    async def _handle_run_once(self, msg: RunOnceMessage):
        if not msg.process_event:
            raise ProcessEventUndefinedException("Must supply a process_event.")
        # Enqueue the event in external buffer
        ext_buf_id = AgentId("external_event_buffer_agent", self.id.key)
        await self._runtime.send_message(
            EnqueueExternalEvent(event_json=msg.process_event.model_dump_json()), ext_buf_id
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
        ext_buf_id = AgentId("external_event_buffer_agent", self.id.key)
        await self._runtime.send_message(
            EnqueueExternalEvent(event_json=msg.process_event.model_dump_json()), ext_buf_id
        )

    async def _handle_get_process_info(self, msg: GetProcessInfoMessage) -> dict:
        if not self.process:
            raise ValueError("Process not set")

        # For each step, if step_agent => ask for step info,
        # if process_agent => ask for process info
        out_steps = []
        for step_id in self.steps:
            if step_id.type == "step_agent":
                s_info = await self._runtime.send_message(ToAutoGenStepInfoMessage(), step_id)
                from semantic_kernel.processes.autogen_runtime.autogen_step_info import AutoGenStepInfo

                out_steps.append(AutoGenStepInfo.model_validate(json.loads(s_info)))
            else:
                # it's a process_agent
                subp_info = await self._runtime.send_message(GetProcessInfoMessage(), step_id)
                from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo

                out_steps.append(AutoGenProcessInfo.model_validate(json.loads(subp_info)))

        # build a new AutoGenProcessInfo

        new_state = KernelProcessState(name=self.process.state.name, id=self.id.key)
        from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo

        new_info = AutoGenProcessInfo(
            inner_step_python_type=self.process.inner_step_python_type,
            state=new_state,
            edges=self.process.edges,
            steps=out_steps,
        )
        return new_info.model_dump_json()

    async def _internal_execute(self, max_supersteps=100, keep_alive=True):
        """The main loop for delivering external events, step messages, etc."""
        for _ in range(max_supersteps):
            if await self._check_end():
                break

            # external events
            await self._handle_external_events()

            # prepare_incoming
            for step_id in self.steps:
                await self._runtime.send_message(PrepareIncomingMessagesMessage(), step_id)

            # check if no messages + not keep_alive
            no_msgs = True
            for step_id in self.steps:
                c = await self._runtime.send_message(CountPreparedMessages(), step_id)
                if c > 0:
                    no_msgs = False
                    break
            if no_msgs and not keep_alive:
                # double check external
                ext_buf_id = AgentId("external_event_buffer_agent", self.id.key)
                leftover = await self._runtime.send_message(DequeueAllExternalEvents(), ext_buf_id)
                if leftover:
                    for ev_str in leftover:
                        await self._runtime.send_message(EnqueueExternalEvent(event_json=ev_str), ext_buf_id)
                    no_msgs = False
                if no_msgs:
                    break

            # process_incoming
            for step_id in self.steps:
                await self._runtime.send_message(ProcessIncomingMessagesMessage(), step_id)

            # handle public events
            await self._handle_public_events()

    async def _check_end(self) -> bool:
        from semantic_kernel.processes.autogen_runtime.messages import DequeueAllMessages

        end_mb_id = AgentId("message_buffer_agent", f"{self.id.key}.{END_PROCESS_ID}")
        leftover = await self._runtime.send_message(DequeueAllMessages(), end_mb_id)
        return bool(leftover)

    async def _handle_external_events(self):
        from semantic_kernel.processes.autogen_runtime.messages import DequeueAllExternalEvents

        ext_buf_id = AgentId("external_event_buffer_agent", self.id.key)
        leftover = await self._runtime.send_message(DequeueAllExternalEvents(), ext_buf_id)

        event_objs = []
        for s in leftover:
            e = json.loads(s)
            ev = KernelProcessEvent.model_validate(e)
            event_objs.append(ev)

        # route each external event to relevant steps
        if not self.output_edges:
            return
        for ev in event_objs:
            if ev.id in self.output_edges:
                for edge in self.output_edges[ev.id]:
                    pm = ProcessMessageFactory.create_from_edge(edge, ev.data)
                    target_step = edge.output_target.step_id
                    if target_step:
                        from semantic_kernel.processes.autogen_runtime.messages import EnqueueMessage

                        mb_id = AgentId("message_buffer_agent", target_step)
                        await self._runtime.send_message(EnqueueMessage(message_json=pm.model_dump_json()), mb_id)

    async def _handle_public_events(self):
        from semantic_kernel.processes.autogen_runtime.messages import DequeueAllEvents

        evbuf_id = AgentId("event_buffer_agent", self.id.key)
        leftover = await self._runtime.send_message(DequeueAllEvents(), evbuf_id)

        for s in leftover:
            loaded = json.loads(s)
            p_ev = ProcessEvent.model_validate(loaded)
            if (
                p_ev.inner_event
                and p_ev.inner_event.visibility == KernelProcessEventVisibility.Public
                and p_ev.id in self.output_edges
            ):
                for edge in self.output_edges[p_ev.id]:
                    pm = ProcessMessageFactory.create_from_edge(edge, p_ev.data)
                    if edge.output_target.step_id:
                        from semantic_kernel.processes.autogen_runtime.messages import EnqueueMessage

                        mb_id = AgentId("message_buffer_agent", edge.output_target.step_id)
                        await self._runtime.send_message(EnqueueMessage(message_json=pm.model_dump_json()), mb_id)
