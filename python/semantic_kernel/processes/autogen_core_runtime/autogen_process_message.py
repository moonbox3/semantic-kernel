# Copyright (c) Microsoft. All rights reserved.

from typing import Any

from semantic_kernel.kernel_pydantic import KernelBaseModel


class AGProcessMessage(KernelBaseModel):
    """Similar to local_runtime.local_message.LocalMessage.

    Used as the AutoGen "payload" object for messaging between step agents in the
    SingleThreadedAgentRuntime.
    """

    source_id: str
    destination_id: str
    function_name: str
    values: dict[str, Any]
    target_event_id: str | None = None
    target_event_data: Any | None = None

    def __str__(self) -> str:
        """Return a string representation of the object."""
        return (
            f"AGProcessMessage("
            f"source_id={self.source_id}, dest_id={self.destination_id}, "
            f"function_name={self.function_name}, values={self.values}, "
            f"target_event_id={self.target_event_id})"
        )
