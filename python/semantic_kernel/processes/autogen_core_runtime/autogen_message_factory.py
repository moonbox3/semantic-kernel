# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING, Any

from semantic_kernel.processes.autogen_core_runtime.autogen_process_message import AGProcessMessage

if TYPE_CHECKING:
    from semantic_kernel.processes.kernel_process.kernel_process_edge import KernelProcessEdge


class AGMessageFactory:
    """A static helper that, given an Edge + data, returns an AGProcessMessage.

    Mimics the local_runtime/local_message_factory.py approach.
    """

    @staticmethod
    def create_from_edge(edge: "KernelProcessEdge", data: Any) -> AGProcessMessage:
        """Create an AGProcessMessage from an Edge and data."""
        target = edge.output_target
        param_value: dict[str, Any] = {}

        if target.parameter_name:
            param_value[target.parameter_name] = data

        return AGProcessMessage(
            source_id=edge.source_step_id,
            destination_id=target.step_id,
            function_name=target.function_name,
            values=param_value,
            target_event_id=target.target_event_id,
            target_event_data=data,
        )
