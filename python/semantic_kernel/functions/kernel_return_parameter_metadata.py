# Copyright (c) Microsoft. All rights reserved.

from typing import Optional, Type

from pydantic import Field

from semantic_kernel.functions.kernel_json_schema import KernelJsonSchema
from semantic_kernel.sk_pydantic import SKBaseModel


class KernelReturnParameterMetadata(SKBaseModel):
    description: str
    parameter_type: Type
    schema: Optional[KernelJsonSchema] = Field(default=None)
