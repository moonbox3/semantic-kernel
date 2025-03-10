# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import logging
from queue import Queue
from typing import Any

from autogen_core import AgentId, BaseAgent, MessageContext

from semantic_kernel.exceptions.process_exceptions import (
    ProcessFunctionNotFoundException,
)
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.autogen_runtime.autogen_step_info import AutoGenStepInfo
from semantic_kernel.processes.autogen_runtime.messages import (
    CountPreparedMessages,
    InitializeStepMessage,
    PrepareIncomingMessagesMessage,
    ProcessIncomingMessagesMessage,
    ToDaprStepInfoMessage,
)
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.process_event import ProcessEvent
from semantic_kernel.processes.process_message import ProcessMessage
from semantic_kernel.processes.process_message_factory import ProcessMessageFactory
from semantic_kernel.processes.step_utils import find_input_channels

logger = logging.getLogger(__name__)


class StepAgent(BaseAgent):
    """A base "step" agent."""

    def __init__(self, agent_id: str, factories: dict[str, Any]):
        """Initialize the StepAgent."""
        super().__init__(agent_id)
        self.factories = factories

        self.parent_process_id: str | None = None
        self.step_info: AutoGenStepInfo | None = None
        self.initialize_task: bool = False
        self.incoming_messages: Queue[ProcessMessage] = Queue()
        self.step_state: KernelProcessStepState | None = None
        self.output_edges: dict[str, Any] = {}
        self.functions: dict[str, KernelFunction] = {}
        self.inputs: dict[str, dict[str, Any]] = {}
        self.initial_inputs: dict[str, dict[str, Any]] = {}
        self.step_activated: bool = False
        self.init_lock = asyncio.Lock()

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """No decorators, so we do manual type-checking."""
        if isinstance(message, InitializeStepMessage):
            return await self._handle_initialize_step(message)
        if isinstance(message, PrepareIncomingMessagesMessage):
            return await self._handle_prepare_incoming_messages()
        if isinstance(message, ProcessIncomingMessagesMessage):
            return await self._handle_process_incoming_messages()
        if isinstance(message, ToDaprStepInfoMessage):
            return await self._handle_to_dapr_step_info()
        if isinstance(message, CountPreparedMessages):
            return self.incoming_messages.qsize()

        logger.warning(f"[StepAgent {self.id}] Unhandled message type: {type(message)}")
        return None

    async def _handle_initialize_step(self, msg: InitializeStepMessage):
        if self.initialize_task:
            return

        payload = json.loads(msg.step_info_json)
        step_info_dict = payload.get("step_info")
        if not step_info_dict:
            raise ValueError("No 'step_info' provided in initialize_step payload")

        # parse AutoGenStepInfo
        self.step_info = AutoGenStepInfo.model_validate(step_info_dict)
        self.step_state = self.step_info.state
        self.output_edges = dict(self.step_info.edges)
        self.parent_process_id = msg.parent_process_id
        self.initialize_task = True

    async def _handle_prepare_incoming_messages(self) -> int:
        from autogen_runtime.messages import DequeueAllMessages

        mb_id = AgentId("message_buffer_agent", self.id.key)
        raw_list = await self.runtime.send_message(DequeueAllMessages(), mb_id)

        count = 0
        for msg_json in raw_list:
            pm = ProcessMessage.model_validate(json.loads(msg_json))
            self.incoming_messages.put(pm)
            count += 1
        return count

    async def _handle_process_incoming_messages(self):
        while not self.incoming_messages.empty():
            pm = self.incoming_messages.get()
            await self._handle_single_message(pm)

    async def _handle_single_message(self, pm: ProcessMessage):
        if not self.step_activated:
            async with self.init_lock:
                if not self.step_activated:
                    await self._activate_step()
                    self.step_activated = True

        # Merge message values into inputs
        if pm.function_name not in self.inputs:
            self.inputs[pm.function_name] = {}
        for k, v in pm.values.items():
            self.inputs[pm.function_name][k] = v

        # see if function is invocable
        can_invoke = True
        for val in self.inputs[pm.function_name].values():
            if val is None:
                can_invoke = False
                break
        if not can_invoke:
            logger.info(f"[{self.id}] function {pm.function_name} missing inputs.")
            return

        fn = self.functions.get(pm.function_name)
        if not fn:
            raise ProcessFunctionNotFoundException(f"Function {pm.function_name} not found in plugin {self.name}")

        kernel = Kernel()
        try:
            invoke_result = await kernel.invoke(fn, **self.inputs[pm.function_name])
            event_name = f"{pm.function_name}.OnResult"
            event_value = invoke_result.value if invoke_result else None
        except Exception as ex:
            logger.error(f"[{self.id}] error: {ex}")
            event_name = f"{pm.function_name}.OnError"
            event_value = str(ex)
        finally:
            from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

            ev = KernelProcessEvent(id=event_name, data=event_value)
            await self._emit_event(ev)

            # reset inputs
            self.inputs[pm.function_name] = self.initial_inputs.get(pm.function_name, {}).copy()

    async def _handle_to_dapr_step_info(self) -> dict:
        if self.step_info and self.step_state:
            self.step_info.state = self.step_state
        return self.step_info.model_dump() if self.step_info else {}

    async def _activate_step(self):
        """Actually create the underlying step object if needed, register plugin functions, etc."""
        if not self.step_info:
            raise ValueError("step_info not set; cannot activate step.")
        # if you have factories for custom step classes:
        if self.factories and (self.step_info.inner_step_python_type in self.factories):
            step_obj = self.factories[self.step_info.inner_step_python_type]()
            if asyncio.iscoroutine(step_obj):
                step_obj = await step_obj
            step_instance: KernelProcessStep = step_obj  # type: ignore
        else:
            # fallback reflection
            parts = self.step_info.inner_step_python_type.rsplit(".", 1)
            mod = __import__(parts[0], fromlist=[parts[1]])
            cls_ = getattr(mod, parts[1])
            step_instance: KernelProcessStep = cls_()  # type: ignore

        kernel = Kernel()
        plugin = kernel.add_plugin(step_instance, self.step_info.state.name)
        for name, fn in plugin.functions.items():
            self.functions[name] = fn

        # set up inputs
        self.initial_inputs = find_input_channels(self, self.functions)
        self.inputs = {k: dict(v) for k, v in self.initial_inputs.items()}

        # run step_instance.activate() if needed
        if self.step_state:
            await step_instance.activate(self.step_state)

    async def _emit_event(self, kp_event: KernelProcessEvent):
        """Emit event to parent's event buffer if public, or local edges."""
        if not self.step_info:
            raise ValueError("Cannot emit event; step_info not set.")

        local_event = ProcessEvent(
            inner_event=kp_event, namespace=f"{self.step_info.state.name}_{self.step_info.state.id}"
        )

        # public => parent's event buffer
        if kp_event.visibility == KernelProcessEventVisibility.Public and self.parent_process_id:
            import json

            from autogen_runtime.messages import EnqueueEvent

            eb_id = AgentId("event_buffer_agent", self.parent_process_id)
            await self.runtime.send_message(EnqueueEvent(event_json=json.dumps(local_event.model_dump())), eb_id)

        # local edges
        import json

        edges = self.output_edges.get(kp_event.id, [])
        for edge in edges:
            pm = ProcessMessageFactory.create_from_edge(edge, kp_event.data)
            if self.parent_process_id:
                mb_id = AgentId("message_buffer_agent", f"{self.parent_process_id}.{edge.output_target.step_id}")
            else:
                mb_id = AgentId("message_buffer_agent", edge.output_target.step_id)
            from autogen_runtime.messages import EnqueueMessage

            msg_json = json.dumps(pm.model_dump())
            await self.runtime.send_message(EnqueueMessage(message_json=msg_json), mb_id)

    @property
    def name(self) -> str:
        """Get the name of the step agent."""
        if not self.step_info:
            return "UninitializedStep"
        return self.step_info.state.name
