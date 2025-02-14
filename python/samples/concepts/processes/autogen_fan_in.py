# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from enum import Enum
from typing import ClassVar

from pydantic import Field

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.processes.autogen_core_runtime.autogen_kernel_process_context import (
    AGKernelProcessContext,
)
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
)
from semantic_kernel.processes.kernel_process.kernel_process_step import KernelProcessStep
from semantic_kernel.processes.kernel_process.kernel_process_step_context import (
    KernelProcessStepContext,
)
from semantic_kernel.processes.kernel_process.kernel_process_step_state import (
    KernelProcessStepState,
)
from semantic_kernel.processes.process_builder import ProcessBuilder

logging.basicConfig(level=logging.WARNING)


class CommonEvents(Enum):
    """Common events for the sample process."""

    UserInputReceived = "UserInputReceived"
    CompletionResponseGenerated = "CompletionResponseGenerated"
    WelcomeDone = "WelcomeDone"
    AStepDone = "AStepDone"
    BStepDone = "BStepDone"
    CStepDone = "CStepDone"
    StartARequested = "StartARequested"
    StartBRequested = "StartBRequested"
    ExitRequested = "ExitRequested"
    StartProcess = "StartProcess"


class KickOffStep(KernelProcessStep):
    KICK_OFF_FUNCTION: ClassVar[str] = "kick_off"

    @kernel_function(name=KICK_OFF_FUNCTION)
    async def print_welcome_message(self, context: KernelProcessStepContext):
        await context.emit_event(process_event=CommonEvents.StartARequested, data="Get Going A")
        await context.emit_event(process_event=CommonEvents.StartBRequested, data="Get Going B")


class AStep(KernelProcessStep):
    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext):
        await asyncio.sleep(1)
        await context.emit_event(process_event=CommonEvents.AStepDone, data="I did A")


class BStep(KernelProcessStep):
    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext):
        await asyncio.sleep(2)
        await context.emit_event(process_event=CommonEvents.BStepDone, data="I did B")


class CStepState:
    current_cycle: int = 0


class CStep(KernelProcessStep[CStepState]):
    state: CStepState = Field(default_factory=CStepState)

    async def activate(self, state: KernelProcessStepState[CStepState]):
        self.state = state.state

    @kernel_function()
    async def do_it(self, context: KernelProcessStepContext, astepdata: str, bstepdata: str):
        self.state.current_cycle += 1
        print(f"CStep Current Cycle: {self.state.current_cycle}")
        if self.state.current_cycle == 3:
            print("CStep Exit Requested")
            await context.emit_event(process_event=CommonEvents.ExitRequested)
            return
        await context.emit_event(process_event=CommonEvents.CStepDone)


async def cycles_with_fan_in_autogen():
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(service_id="default"))

    process = ProcessBuilder(name="Test Process")
    kickoff_step = process.add_step(step_type=KickOffStep)
    myAStep = process.add_step(step_type=AStep)
    myBStep = process.add_step(step_type=BStep)
    myCStep = process.add_step(step_type=CStep)

    process.on_input_event(CommonEvents.StartProcess).send_event_to(kickoff_step)
    kickoff_step.on_event(CommonEvents.StartARequested).send_event_to(myAStep)
    kickoff_step.on_event(CommonEvents.StartBRequested).send_event_to(myBStep)
    myAStep.on_event(CommonEvents.AStepDone).send_event_to(myCStep, parameter_name="astepdata")
    myBStep.on_event(CommonEvents.BStepDone).send_event_to(myCStep, parameter_name="bstepdata")
    myCStep.on_event(CommonEvents.CStepDone).send_event_to(kickoff_step)
    myCStep.on_event(CommonEvents.ExitRequested).stop_process()

    kernel_process = process.build()

    async with AGKernelProcessContext(kernel_process, kernel) as process_context:
        await process_context.start_with_event(KernelProcessEvent(id=CommonEvents.StartProcess.value, data="foo"))

        process_state: KernelProcess = await process_context.get_state()
        c_step_info = next((s for s in process_state.steps if s.state.name == "CStep"), None)
        if c_step_info:
            c_state_obj = c_step_info.state
            if c_state_obj and c_state_obj.state:
                print(f"Final State Check: CStepState current cycle: {c_state_obj.state.current_cycle}")
        else:
            print("No CStep found in final state.")


if __name__ == "__main__":
    asyncio.run(cycles_with_fan_in_autogen())
