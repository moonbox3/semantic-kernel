# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import uuid
from queue import Queue
from typing import TYPE_CHECKING, Any

from autogen_core import BaseAgent, MessageContext

from semantic_kernel.exceptions.kernel_exceptions import KernelException
from semantic_kernel.exceptions.process_exceptions import (
    ProcessFunctionNotFoundException,
    ProcessTargetFunctionNameMismatchException,
)
from semantic_kernel.processes.autogen_core_runtime.autogen_process_message import AGProcessMessage
from semantic_kernel.processes.kernel_process.kernel_process_edge import KernelProcessEdge
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
)
from semantic_kernel.processes.kernel_process.kernel_process_message_channel import KernelProcessMessageChannel
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_info import KernelProcessStepInfo
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.local_runtime.local_event import LocalEvent
from semantic_kernel.processes.process_types import get_generic_state_type
from semantic_kernel.processes.step_utils import find_input_channels

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from semantic_kernel.functions.kernel_function import KernelFunction
    from semantic_kernel.kernel import Kernel


class AGStepAgent(KernelProcessMessageChannel, BaseAgent):
    """Each step in the process is an Agent in the SingleThreadedAgentRuntime.

    We route incoming AGProcessMessages to the function call, produce results,
    and emit events if needed.
    """

    def __init__(
        self,
        description: str,
        kernel: "Kernel",
        step_info: KernelProcessStepInfo,
        parent_process_id: str | None = None,
    ):
        """Initialize the step agent with the kernel, step info, etc."""
        super().__init__(description)
        self.kernel = kernel
        self.step_info = step_info
        self.step_state: KernelProcessStepState = step_info.state
        self.id_str: str = step_info.state.id or uuid.uuid4().hex
        self.parent_process_id = parent_process_id
        self.output_edges = step_info.output_edges

        self.functions: dict[str, Any] = {}
        self.inputs: dict[str, dict[str, Any]] = {}
        self.initial_inputs: dict[str, dict[str, Any]] = {}

        self.step_activated = False
        self.activate_lock = asyncio.Lock()

        self.outgoing_event_queue: Queue[LocalEvent] = Queue()
        self.init_lock: asyncio.Lock = asyncio.Lock()
        self.initialize_task: bool | None = False
        self.event_namespace = f"{step_info.state.name}_{step_info.state.id}"

    @property
    def name(self) -> str:
        """Name of the step from step_info."""
        return self.step_info.state.name

    @property
    def type(self) -> str:
        """Agent type for SingleThreadedAgentRuntime registry."""
        return f"ag_step_{self.id_str}"

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        """Called whenever we get a message.

        If it's an AGProcessMessage with the correct function
        name, we see if inputs are ready to run. Then invoke the kernel function if so.
        """
        if isinstance(message, AGProcessMessage):
            return await self.handle_message(message)
        return None

    async def handle_message(self, message: AGProcessMessage):
        """Handles a LocalMessage that has been sent to the step."""
        if message is None:
            raise ValueError("The message is None.")

        if not self.initialize_task:
            async with self.init_lock:
                # Second check to ensure that initialization happens only once
                # This avoids a race condition where multiple coroutines might
                # reach the first check at the same time before any of them acquire the lock.
                if not self.initialize_task:
                    await self.initialize_step()
                    self.initialize_task = True

        if self.functions is None or self.inputs is None or self.initial_inputs is None:
            raise ValueError("The step has not been initialized.")

        message_log_parameters = ", ".join(f"{k}: {v}" for k, v in message.values.items())
        logger.info(
            f"Received message from `{message.source_id}` targeting function "
            f"`{message.function_name}` and parameters `{message_log_parameters}`."
        )

        # Add the message values to the inputs for the function
        for k, v in message.values.items():
            if self.inputs.get(message.function_name) and self.inputs[message.function_name].get(k):
                logger.info(
                    f"Step {self.name} already has input for `{message.function_name}.{k}`, "
                    f"it is being overwritten with a message from Step named `{message.source_id}`."
                )

            if message.function_name not in self.inputs:
                self.inputs[message.function_name] = {}

            self.inputs[message.function_name][k] = v

        invocable_functions = [
            k
            for k, v in self.inputs.items()
            if v is not None and (v == {} or all(val is not None for val in v.values()))
        ]
        missing_keys = [
            f"{outer_key}.{inner_key}"
            for outer_key, outer_value in self.inputs.items()
            for inner_key, inner_value in outer_value.items()
            if inner_value is None
        ]

        if not invocable_functions:
            logger.info(f"No invocable functions, missing keys: {', '.join(missing_keys)}")
            return

        target_function = next((name for name in invocable_functions if name == message.function_name), None)

        if not target_function:
            raise ProcessTargetFunctionNameMismatchException(
                f"A message targeting function `{message.function_name}` has resulted in a different function "
                f"`{invocable_functions[0]}` becoming invocable. Check the function names."
            )

        logger.info(
            f"Step with Id '{self.id}' received all required input for function [{target_function}] and is executing."
        )

        # Concatenate all inputs and run the function
        arguments = self.inputs[target_function]
        function = self.functions.get(target_function)

        if function is None:
            raise ProcessFunctionNotFoundException(f"Function {target_function} not found in plugin {self.name}")

        invoke_result = None
        event_name = None
        event_value = None

        try:
            logger.info(
                f"Invoking plugin `{function.plugin_name}` and function `{function.name}` with arguments: {arguments}"
            )
            invoke_result = await self.invoke_function(function, self.kernel, arguments)
            event_name = f"{target_function}.OnResult"
            event_value = invoke_result.value
        except Exception as ex:
            logger.error(f"Error in Step {self.name}: {ex!s}")
            event_name = f"{target_function}.OnError"
            event_value = str(ex)
        finally:
            await self.emit_event(KernelProcessEvent(id=event_name, data=event_value))

            # Reset the inputs for the function that was just executed
            self.inputs[target_function] = self.initial_inputs.get(target_function, {}).copy()

    async def invoke_function(self, function: "KernelFunction", kernel: "Kernel", arguments: dict[str, Any]):
        """Invokes the function."""
        return await kernel.invoke(function, **arguments)

    async def emit_event(self, process_event: KernelProcessEvent):
        """Emits an event from the step."""
        await self.emit_local_event(LocalEvent.from_kernel_process_event(process_event, self.event_namespace))

    async def emit_local_event(self, local_event: "LocalEvent"):
        """Emits an event from the step."""
        scoped_event = self.scoped_event(local_event)
        self.outgoing_event_queue.put(scoped_event)

    async def initialize_step(self):
        """Initializes the step."""
        # Instantiate an instance of the inner step object
        step_cls = self.step_info.inner_step_type

        step_instance: KernelProcessStep = step_cls()  # type: ignore

        kernel_plugin = self.kernel.add_plugin(
            step_instance, self.step_info.state.name if self.step_info.state else "default_name"
        )

        # Load the kernel functions
        for name, f in kernel_plugin.functions.items():
            self.functions[name] = f

        # Initialize the input channels
        self.initial_inputs = find_input_channels(channel=self, functions=self.functions)
        self.inputs = {k: {kk: vv for kk, vv in v.items()} if v else {} for k, v in self.initial_inputs.items()}

        # Use the existing state or create a new one if not provided
        state_object = self.step_info.state

        # Extract TState from inner_step_type
        t_state = get_generic_state_type(step_cls)

        if t_state is not None:
            # Create state_type as KernelProcessStepState[TState]
            state_type = KernelProcessStepState[t_state]

            if state_object is None:
                state_object = state_type(
                    name=step_cls.__name__,
                    id=step_cls.__name__,
                    state=None,
                )
            else:
                # Make sure state_object is an instance of state_type
                if not isinstance(state_object, KernelProcessStepState):
                    error_message = "State object is not of the expected type."
                    raise KernelException(error_message)

            # Make sure that state_object.state is not None
            if state_object.state is None:
                try:
                    state_object.state = t_state()
                except Exception as e:
                    error_message = f"Cannot instantiate state of type {t_state}: {e}"
                    raise KernelException(error_message)
        else:
            # The step has no user-defined state; use the base KernelProcessStepState
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

        # Set the step state and activate the step with the state object
        self.step_state = state_object
        await step_instance.activate(state_object)

    def get_edge_for_event(self, event_id: str) -> list[KernelProcessEdge]:
        """Retrieves all edges that are associated with the provided event Id."""
        if not self.output_edges:
            return []

        return self.output_edges.get(event_id, [])

    def get_all_events(self) -> list["LocalEvent"]:
        """Retrieves all events that have been emitted by this step in the previous superstep."""
        all_events = []
        while not self.outgoing_event_queue.empty():
            all_events.append(self.outgoing_event_queue.get())
        return all_events

    def scoped_event(self, local_event: "LocalEvent") -> "LocalEvent":
        """Generates a scoped event for the step."""
        if local_event is None:
            raise ValueError("The local event must be specified.")
        local_event.namespace = f"{self.name}_{self.id}"
        return local_event

    def scoped_event_from_kernel_process(self, process_event: "KernelProcessEvent") -> "LocalEvent":
        """Generates a scoped event for the step from a KernelProcessEvent."""
        if process_event is None:
            raise ValueError("The process event must be specified.")
        return LocalEvent.from_kernel_process_event(process_event, f"{self.name}_{self.id}")
