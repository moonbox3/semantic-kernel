# Copyright (c) Microsoft. All rights reserved.

from typing import Optional, Type

from pydantic import Field

from semantic_kernel.functions.kernel_json_schema import KernelJsonSchema
from semantic_kernel.kernel_pydantic import KernelBaseModel


class KernelReturnParameterMetadata(KernelBaseModel):
    # TODO: Placeholder for upcoming work on replacing context, introducing Kernel Args

    description: str
    parameter_type: Type
    schema: Optional[KernelJsonSchema] = Field(default=None)
