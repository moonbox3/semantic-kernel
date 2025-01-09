# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import uuid
from typing import Any

from autogen_core import AgentId, BaseAgent, MessageContext

from semantic_kernel.exceptions.kernel_exceptions import KernelException
from semantic_kernel.exceptions.process_exceptions import (
    ProcessFunctionNotFoundException,
    ProcessTargetFunctionNameMismatchException,
)
from semantic_kernel.kernel import Kernel
from semantic_kernel.processes.autogen_core_runtime.autogen_message_factory import AGMessageFactory
from semantic_kernel.processes.autogen_core_runtime.autogen_process_message import AGProcessMessage
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_info import KernelProcessStepInfo
from semantic_kernel.processes.kernel_process.kernel_process_step_state import KernelProcessStepState
from semantic_kernel.processes.process_types import get_generic_state_type

logger: logging.Logger = logging.getLogger(__name__)


class AGStepAgent(BaseAgent):
    """Each "step" in the process is an Agent in the SingleThreadedAgentRuntime.

    In local_runtime, we had a local Step object. Here, we replicate that
    with an AutoGen-based agent that receives AGProcessMessages, executes
    the kernel function, and emits events.
    """

    def __init__(
        self,
        description: str,
        kernel: Kernel,
        step_info: KernelProcessStepInfo,
        parent_process_id: str | None = None,
    ):
        """Initialize the step agent with the kernel, step info, and parent process ID."""
        super().__init__(description)

        self.kernel = kernel
        self.step_info = step_info
        self.step_state: KernelProcessStepState = step_info.state
        self.id_str: str = step_info.state.id if step_info.state.id is not None else uuid.uuid4().hex
        self.parent_process_id = parent_process_id
        self.output_edges = step_info.output_edges
        # track function definitions
        self.functions: dict[str, Any] = {}
        # track inputs
        self.inputs: dict[str, dict[str, Any]] = {}
        self.initial_inputs: dict[str, dict[str, Any]] = {}
        # whether this agent's "step" is activated
        self.step_activated = False
        self.activate_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """Name of the step from step_info."""
        return self.step_info.state.name

    @property
    def type(self) -> str:
        """Agent type is 'ag_step' + the step's name, for debugging."""
        return f"ag_step_{self.id_str}"

    async def on_message_impl(self, message: Any, ctx: MessageContext) -> Any:
        """Called whenever the runtime sends us a message of any recognized type.

        We handle:
          - AGProcessMessage: run the function call if inputs are complete
        """
        if isinstance(message, AGProcessMessage):
            return await self.handle_process_message(message)
        return None

    async def handle_process_message(self, message: AGProcessMessage):
        """Main logic to handle an incoming AGProcessMessage for the step."""
        if not self.step_activated:
            async with self.activate_lock:
                if not self.step_activated:
                    await self._activate_step()
                    self.step_activated = True

        # Merge inbound values into self.inputs
        for k, v in message.values.items():
            if message.function_name not in self.inputs:
                self.inputs[message.function_name] = {}
            self.inputs[message.function_name][k] = v

        # Check if all required inputs are satisfied
        invocable_functions = []
        for fname, param_dict in self.inputs.items():
            if param_dict is None:
                continue
            # check if everything is provided
            if all(val is not None for val in param_dict.values()):
                invocable_functions.append(fname)

        if message.function_name not in invocable_functions:
            # possibly missing inputs
            # or mismatch
            # raise if mismatch
            if invocable_functions:
                raise ProcessTargetFunctionNameMismatchException(
                    f"A message targeting function `{message.function_name}` "
                    f" ended up with a different invocable function {invocable_functions[0]}."
                )
            return  # missing some inputs => no action

        # everything is ready => run
        target_function = message.function_name
        function_obj = self.functions.get(target_function)
        if not function_obj:
            raise ProcessFunctionNotFoundException(f"Function {target_function} not found in plugin {self.name}")

        # gather arguments
        arguments = self.inputs[target_function]
        event_name = f"{target_function}.OnResult"
        event_value = None
        try:
            # run the function
            logger.info(f"Invoking function {target_function} with arguments {arguments}")
            invoke_result = await self.kernel.invoke(function_obj, **arguments)
            event_value = invoke_result.value
        except Exception as e:
            logger.error(f"Error in step {self.name}: {e}")
            event_name = f"{target_function}.OnError"
            event_value = str(e)
        finally:
            await self._emit_kernel_event(
                KernelProcessEvent(id=event_name, data=event_value, visibility=KernelProcessEventVisibility.Internal)
            )
            # reset function inputs
            self.inputs[target_function] = self.initial_inputs.get(target_function, {}).copy()

        # if there's a target event ID, we might nest an event for it
        if message.target_event_id is not None:
            pass

    async def _emit_kernel_event(self, proc_event: KernelProcessEvent):
        """Emit an event from this step. Possibly triggers edges to next step."""
        # We'll treat the event like a "local" event => For every matching edge, create AGProcessMessage
        event_id = proc_event.id
        matching_edges = self.output_edges.get(event_id, [])
        for edge in matching_edges:
            msg = AGMessageFactory.create_from_edge(edge, proc_event.data)
            # send to the next step
            # We'll send a direct message to the next step agent
            next_step_id = AgentId(f"ag_step_{edge.output_target.step_id}", edge.output_target.step_id)
            await self.send_message(msg, recipient=next_step_id)

        # if this event is "Public" and there's a parent process, that might mean
        # we publish a "ProcessEvent"?

    async def _activate_step(self):
        """Load up the plugin for the step's inner class, set up the input schema."""
        step_cls = self.step_info.inner_step_type
        step_instance: KernelProcessStep = step_cls()
        plugin_name = self.step_info.state.name or "default_name"
        plugin_funcs = self.kernel.add_plugin(step_instance, plugin_name)
        self.functions = dict(plugin_funcs.functions)

        # figure out each function's required arguments
        # see local_runtime for "find_input_channels"
        # we'll do a simplified approach:
        for fname, func_obj in self.functions.items():
            # gather parameter names
            param_dict = {}
            for p in func_obj.metadata.parameters:
                # skip "KernelProcessStepContext" or "Kernel"
                if p.type_ not in ("KernelProcessStepContext", "Kernel"):
                    param_dict[p.name] = None
            self.inputs[fname] = param_dict
            self.initial_inputs[fname] = dict(param_dict)

        # next, if the step class is a KernelProcessStep[TState], handle state
        t_state = get_generic_state_type(step_cls)
        if t_state is not None and not self.step_state.state:
            # instantiate TState
            try:
                self.step_state.state = t_state()
            except Exception as e:
                raise KernelException(f"Cannot instantiate state of type {t_state}: {e}")
