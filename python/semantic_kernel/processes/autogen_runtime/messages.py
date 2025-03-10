# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass

from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

#
# ---- Step Agent messages ----
#


@dataclass
class InitializeStepMessage:
    """Tells a StepAgent to initialize itself with a DaprStepInfo-like payload in JSON form."""

    step_info_json: str
    parent_process_id: str | None = None


@dataclass
class PrepareIncomingMessagesMessage:
    """Tells a StepAgent to dequeue all pending messages from its local queue.

    Returns the count of retrieved messages.
    """

    pass


@dataclass
class ProcessIncomingMessagesMessage:
    """Tells a StepAgent to process everything it has queued up."""

    pass


@dataclass
class ToDaprStepInfoMessage:
    """Asks a StepAgent to return a `DaprStepInfo` describing the current state of this step."""

    pass


@dataclass
class CountPreparedMessages:
    """For debugging. StepAgent: how many messages do you have queued up?"""

    pass


#
# ---- Process Agent messages ----
#


@dataclass
class InitializeProcessMessage:
    """Tells a ProcessAgent to initialize with a DaprProcessInfo plus optional parent ID."""

    process_info: AutoGenProcessInfo
    parent_process_id: str | None = None


@dataclass
class StartProcessMessage:
    """Tells a ProcessAgent to start running in the background."""

    keep_alive: bool = True


@dataclass
class RunOnceMessage:
    """Tells a ProcessAgent to accept a KernelProcessEvent once, run the loop, and then stop."""

    process_event: KernelProcessEvent


@dataclass
class StopProcessMessage:
    """Tells a ProcessAgent to stop."""

    pass


@dataclass
class SendProcessMessage:
    """Tells a ProcessAgent to queue a KernelProcessEvent without forcibly starting."""

    process_event: KernelProcessEvent


@dataclass
class GetProcessInfoMessage:
    """Asks a ProcessAgent to return a dict describing the current process state."""

    pass


#
# ---- Shared step <-> process messages, if you want them ----
#


@dataclass
class EnqueueEvent:
    """For event buffer, Please enqueue this event JSON."""

    event_json: str


@dataclass
class DequeueAllEvents:
    """For event buffer. Dequeue all events, return them as a list."""

    pass


@dataclass
class EnqueueExternalEvent:
    """For external event buffer. Please enqueue this external event JSON."""

    event_json: str


@dataclass
class DequeueAllExternalEvents:
    """For external event buffer."""

    pass


@dataclass
class EnqueueMessage:
    """For message buffer. Enqueue a message string."""

    message_json: str


@dataclass
class DequeueAllMessages:
    """For message buffer. Dequeue all messages, returning them."""

    pass
