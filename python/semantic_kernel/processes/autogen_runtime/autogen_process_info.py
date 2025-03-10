# Copyright (c) Microsoft. All rights reserved.

from typing import Literal, MutableSequence

from pydantic import Field

from semantic_kernel.processes.autogen_runtime.autogen_step_info import AutoGenStepInfo
from semantic_kernel.processes.kernel_process.kernel_process import KernelProcess
from semantic_kernel.processes.kernel_process.kernel_process_state import KernelProcessState
from semantic_kernel.utils.feature_stage_decorator import experimental


@experimental
class AutoGenProcessInfo(AutoGenStepInfo):
    """A direct analog to DaprProcessInfo, but renamed for clarity in AutoGen context."""

    type: Literal["AutoGenProcessInfo"] = "AutoGenProcessInfo"
    steps: MutableSequence["AutoGenStepInfo | AutoGenProcessInfo"] = Field(default_factory=list)

    def to_kernel_process(self) -> KernelProcess:
        """Converts the AutoGenProcessInfo to a KernelProcess."""
        if not isinstance(self.state, KernelProcessState):
            raise ValueError("State must be a kernel process state")

        steps = []
        for step in self.steps:
            steps.append(step.to_kernel_process_step_info())

        return KernelProcess(state=self.state, steps=steps, edges=self.edges)

    @classmethod
    def from_kernel_process(cls, kernel_process: KernelProcess) -> "AutoGenProcessInfo":
        """Converts a KernelProcess to an AutoGenProcessInfo."""
        if kernel_process is None:
            raise ValueError("Kernel process must be provided")

        base_info = AutoGenStepInfo.from_kernel_step_info(kernel_process)
        sub_steps = []

        for step in kernel_process.steps:
            if isinstance(step, KernelProcess):
                # Wrap it with AutoGenProcessInfo
                sub_steps.append(cls.from_kernel_process(step))
            else:
                sub_steps.append(AutoGenStepInfo.from_kernel_step_info(step))

        return AutoGenProcessInfo(
            inner_step_python_type=base_info.inner_step_python_type,
            state=base_info.state,
            edges={k: list(v) for k, v in base_info.edges.items()},
            steps=sub_steps,
        )
