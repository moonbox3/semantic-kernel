# Copyright (c) Microsoft. All rights reserved.

import asyncio
import importlib
import json
import logging
from inspect import isawaitable
from queue import Queue
from typing import TYPE_CHECKING, Any

from agent_runtime import AgentId, BaseAgent, CoreAgentId, MessageContext

from semantic_kernel.exceptions.kernel_exceptions import KernelException
from semantic_kernel.exceptions.process_exceptions import (
    ProcessFunctionNotFoundException,
)
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.core_runtime.core_step_info import CoreStepInfo
from semantic_kernel.processes.core_runtime.messages import (
    EnqueueEvent,
    EnqueueMessage,
    InitializeStepMessage,
    PrepareIncomingMessagesMessage,
    ProcessIncomingMessagesMessage,
    ToCoreStepInfoMessage,
)
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)
from semantic_kernel.processes.kernel_process.kernel_process_message_channel import KernelProcessMessageChannel
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.process_event import ProcessEvent
from semantic_kernel.processes.process_message import ProcessMessage
from semantic_kernel.processes.process_message_factory import ProcessMessageFactory
from semantic_kernel.processes.process_types import get_generic_state_type
from semantic_kernel.processes.step_utils import find_input_channels

if TYPE_CHECKING:
    from semantic_kernel.kernel import Kernel
    from semantic_kernel.processes.kernel_process.kernel_process_edge import KernelProcessEdge

logger = logging.getLogger(__name__)


class CoreStep(BaseAgent, KernelProcessMessageChannel):
    """A base "step" agent."""

    def __init__(self, agent_id: AgentId, kernel: "Kernel", factories: dict[str, Any], runtime):
        """Initialize the StepAgent."""
        super().__init__(agent_id.key)
        self._id = agent_id
        self._runtime = runtime
        self.kernel = kernel
        self.factories = factories

        self.event_namespace: str | None = None
        self.parent_process_id: str | None = None
        self.step_info: CoreStepInfo | None = None
        self.initialize_task: bool = False
        self.inner_step_type: str | None = None
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
        if isinstance(message, ToCoreStepInfoMessage):
            return await self._handle_to_process_step_info()

        logger.warning(f"[StepAgent {self._id.key}] Unhandled message type: {type(message)}")
        raise ValueError(f"Unhandled message type: {type(message)}")

    async def _handle_initialize_step(self, msg: InitializeStepMessage):
        if self.initialize_task:
            return

        step_info_dict = json.loads(msg.step_info_str)
        self.step_info = CoreStepInfo.model_validate(step_info_dict)
        self.inner_step_type = self.step_info.inner_step_python_type

        self.step_state = self.step_info.state
        self.output_edges = {k: v for k, v in self.step_info.edges.items()}
        self.parent_process_id = msg.parent_process_id
        self.event_namespace = f"{self.step_info.state.name}_{self.step_info.state.id}"
        self.initialize_task = True

    async def _handle_prepare_incoming_messages(self) -> int:
        from semantic_kernel.processes.core_runtime.messages import DequeueAllMessages

        mb_id = CoreAgentId("MessageBufferAgent", f"{self._id.key}.{self.step_state.id}")
        raw_list = await self._runtime.send_message(DequeueAllMessages(), mb_id)

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
            logger.info(f"[{self._id.key}] function {pm.function_name} missing inputs.")
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
            await self.emit_event(ev)

            # reset inputs
            self.inputs[pm.function_name] = self.initial_inputs.get(pm.function_name, {}).copy()

    async def _handle_to_process_step_info(self) -> dict:
        """Converts the step to a CoreStepInfo object."""
        if not self.step_activated:
            async with self.init_lock:
                # Second check to ensure that initialization happens only once
                # This avoids a race condition where multiple coroutines might
                # reach the first check at the same time before any of them acquire the lock.
                if not self.step_activated:
                    await self._activate_step()
                    self.step_activated = True

        if self.step_info is None:
            raise ValueError("The step must be initialized before converting to CoreStepInfo.")

        if self.inner_step_type is None:
            raise ValueError("The inner step type must be initialized before converting to CoreStepInfo.")

        if self.step_state is not None:
            self.step_info.state = self.step_state

        step_info = CoreStepInfo(
            inner_step_python_type=self.inner_step_type,
            state=self.step_info.state,
            edges=self.step_info.edges,
        )

        return step_info.model_dump()

    def _get_class_from_string(self, full_class_name: str):
        """Gets a class from a string."""
        module_name, class_name = full_class_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    async def _activate_step(self):
        # Instantiate an instance of the inner step object and retrieve its class reference.
        if self.factories and self.inner_step_type in self.factories:
            step_object = self.factories[self.inner_step_type]()
            if isawaitable(step_object):
                step_object = await step_object
            step_cls = step_object.__class__
            step_instance: KernelProcessStep = step_object  # type: ignore
        else:
            step_cls = self._get_class_from_string(self.inner_step_type)
            step_instance: KernelProcessStep = step_cls()  # type: ignore

        kernel_plugin = self.kernel.add_plugin(
            step_instance,
            self.step_info.state.name if self.step_info.state else "default_name",
        )

        # Load the kernel functions.
        for name, f in kernel_plugin.functions.items():
            self.functions[name] = f

            # TODO(evmattso): handle creating the external process channel actor used for external messaging

        # Initialize the input channels.
        self.initial_inputs = find_input_channels(channel=self, functions=self.functions)
        self.inputs = {k: {kk: vv for kk, vv in v.items()} if v else {} for k, v in self.initial_inputs.items()}

        # Use the existing state or create a new one if not provided.
        state_object = self.step_info.state

        # Extract TState from inner_step_type using the class reference.
        t_state = get_generic_state_type(step_cls)

        if t_state is not None:
            state_type = KernelProcessStepState[t_state]

            if state_object is None:
                # Create a fresh step state object if none is provided.
                state_object = state_type(
                    name=step_cls.__name__,
                    id=step_cls.__name__,
                    state=None,
                )
            else:
                # Ensure that state_object is an instance of the expected type.
                if not isinstance(state_object, KernelProcessStepState):
                    error_message = "State object is not of the expected KernelProcessStepState type."
                    raise KernelException(error_message)

            # If state is None, instantiate it. If it exists but is not the right type, validate it.
            if state_object.state is None:
                try:
                    state_object.state = t_state()
                except Exception as e:
                    error_message = f"Cannot instantiate state of type {t_state}: {e}"
                    raise KernelException(error_message)
            else:
                # Convert the existing state if it's not already an instance of t_state
                if not isinstance(state_object.state, t_state):
                    try:
                        state_object.state = t_state.model_validate(state_object.state)
                    except Exception as e:
                        error_message = f"Cannot validate state of type {t_state}: {e}"
                        raise KernelException(error_message)
        else:
            # The step has no user-defined state; use the base KernelProcessStepState.
            state_type = KernelProcessStepState
            if state_object is None:
                state_object = state_type(
                    name=step_cls.__name__,
                    id=step_cls.__name__,
                    state=None,
                )

        if state_object is None:
            error_message = "The state object for the KernelProcessStep could not be created."
            raise KernelException(error_message)

        # Set the step state and activate the step with the state object.
        self.step_state = state_object
        await step_instance.activate(state_object)

    async def emit_event(self, process_event: KernelProcessEvent):
        """Emits an event from the step."""
        if self.event_namespace is None:
            raise ValueError("The event namespace must be initialized before emitting an event.")

        await self.emit_process_event(ProcessEvent(inner_event=process_event, namespace=self.event_namespace))

    async def emit_process_event(self, core_event: ProcessEvent):
        """Emits an event from the step."""
        if core_event.visibility == KernelProcessEventVisibility.Public and self.parent_process_id is not None:
            parent_process_id = CoreAgentId("EventBufferAgent", self.parent_process_id)
            await self._runtime.send_message(EnqueueEvent(event_json=core_event.model_dump_json()), parent_process_id)

        for edge in self.get_edge_for_event(core_event.id):
            message: ProcessMessage = ProcessMessageFactory.create_from_edge(edge, core_event.data)
            scoped_step_id = self._scoped_actor_id(CoreAgentId("MessageBufferAgent", edge.output_target.step_id))
            await self._runtime.send_message(EnqueueMessage(message_json=message.model_dump_json()), scoped_step_id)

        # TODO(evmattso): handle cases where error event is raised

    def _scoped_actor_id(self, agent_id: AgentId, scope_to_parent: bool = False) -> str:
        """Returns the scoped actor ID for the given actor ID."""
        if scope_to_parent and self.parent_process_id is None:
            raise ValueError("Cannot scope to parent process ID when it is None.")

        id = self.parent_process_id if scope_to_parent else self._id.key

        return CoreAgentId(agent_id.type, f"{id}.{agent_id.key}")

    @property
    def name(self) -> str:
        """Get the name of the step agent."""
        if not self.step_info:
            return "UninitializedStep"
        return self.step_info.state.name

    def get_edge_for_event(self, event_id: str) -> list["KernelProcessEdge"]:
        """Retrieves all edges that are associated with the provided event Id."""
        if not self.output_edges:
            return []

        return self.output_edges.get(event_id, [])
