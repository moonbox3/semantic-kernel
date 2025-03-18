# Copyright (c) Microsoft. All rights reserved.


from semantic_kernel.kernel_pydantic import KernelBaseModel


class TaskStartMessage(KernelBaseModel):
    """Message to start a task."""

    body: str
