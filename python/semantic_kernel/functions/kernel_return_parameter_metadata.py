# Copyright (c) Microsoft. All rights reserved.

from typing import Any, ClassVar, Dict, List, Optional, Type, Union

from pydantic import Field, root_validator
from semantic_kernel.sk_pydantic import SKBaseModel

from semantic_kernel.functions.kernel_json_schema import KernelJsonSchema


class KernelReturnParameterMetadata(SKBaseModel):
    description: str
    parameter_type: Type
    schema: Optional[KernelJsonSchema] = Field(default=None)
