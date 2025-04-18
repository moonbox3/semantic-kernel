# Copyright (c) Microsoft. All rights reserved.

from typing import Any

from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.processes.kernel_process.kernel_process_event import (
    KernelProcessEvent,
    KernelProcessEventVisibility,
)


class AGProcessEvent(KernelBaseModel):
    """Similar to local_runtime.local_event.LocalEvent.

    Representing an event inside the SingleThreadedAgentRuntime environment.
    This can be published or directed to a "process" agent for orchestration.
    """

    namespace: str | None = None
    inner_event: KernelProcessEvent

    @property
    def id(self) -> str:
        """Unique ID for the event, e.g. 'MyStep_1234.OnResult'."""
        if not self.namespace:
            return self.inner_event.id
        return f"{self.namespace}.{self.inner_event.id}"

    @property
    def data(self) -> Any:
        """Data associated with the event."""
        return self.inner_event.data

    @property
    def visibility(self) -> KernelProcessEventVisibility:
        """Visibility of the event."""
        return self.inner_event.visibility
