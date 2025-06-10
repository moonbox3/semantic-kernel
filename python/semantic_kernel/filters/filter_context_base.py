# Copyright (c) Microsoft. All rights reserved.

from typing import TYPE_CHECKING

from semantic_kernel.kernel_pydantic import KernelBaseModel

if TYPE_CHECKING:
    pass


class FilterContextBase(KernelBaseModel):
    """Base class for Kernel Filter Contexts."""

    # Temp removal for POC
    is_streaming: bool = False
