# Copyright (c) Microsoft. All rights reserved.


from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.processes.autogen_runtime.autogen_process_info import AutoGenProcessInfo
from semantic_kernel.processes.kernel_process.kernel_process_event import KernelProcessEvent

#
# ---- Step Agent messages ----
#


class InitializeStepMessage(KernelBaseModel):
    """Tells a StepAgent to initialize itself with a AutoGenStepInfo payload in JSON form."""

    step_info_json: str
    parent_process_id: str | None = None


class PrepareIncomingMessagesMessage(KernelBaseModel):
    """Tells a StepAgent to dequeue all pending messages from its local queue.

    Returns the count of retrieved messages.
    """

    pass


class ProcessIncomingMessagesMessage(KernelBaseModel):
    """Tells a StepAgent to process everything it has queued up."""

    pass


class ToAutoGenStepInfoMessage(KernelBaseModel):
    """Asks a StepAgent to return a `AutoGenStepInfo` describing the current state of this step."""

    pass


class CountPreparedMessages(KernelBaseModel):
    """For debugging. StepAgent: how many messages do you have queued up?"""

    pass


#
# ---- Process Agent messages ----
#


class InitializeProcessMessage(KernelBaseModel):
    """Tells a ProcessAgent to initialize with a DaprProcessInfo plus optional parent ID."""

    process_info: AutoGenProcessInfo
    parent_process_id: str | None = None


class StartProcessMessage(KernelBaseModel):
    """Tells a ProcessAgent to start running in the background."""

    keep_alive: bool = True


class RunOnceMessage(KernelBaseModel):
    """Tells a ProcessAgent to accept a KernelProcessEvent once, run the loop, and then stop."""

    process_event: KernelProcessEvent


class StopProcessMessage(KernelBaseModel):
    """Tells a ProcessAgent to stop."""

    pass


class SendProcessMessage(KernelBaseModel):
    """Tells a ProcessAgent to queue a KernelProcessEvent without forcibly starting."""

    process_event: KernelProcessEvent


class GetProcessInfoMessage(KernelBaseModel):
    """Asks a ProcessAgent to return a dict describing the current process state."""

    pass


#
# ---- Shared step <-> process messages, if you want them ----
#


class EnqueueEvent(KernelBaseModel):
    """For event buffer, Please enqueue this event JSON."""

    event_json: str


class DequeueAllEvents(KernelBaseModel):
    """For event buffer. Dequeue all events, return them as a list."""

    pass


class EnqueueExternalEvent(KernelBaseModel):
    """For external event buffer. Please enqueue this external event JSON."""

    event_json: str


class DequeueAllExternalEvents(KernelBaseModel):
    """For external event buffer."""

    pass


class EnqueueMessage(KernelBaseModel):
    """For message buffer. Enqueue a message string."""

    message_json: str


class DequeueAllMessages(KernelBaseModel):
    """For message buffer. Dequeue all messages, returning them."""

    pass
