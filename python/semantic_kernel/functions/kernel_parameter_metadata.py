# Copyright (c) Microsoft. All rights reserved.

from typing import Any, Optional, Type

from pydantic import Field

from semantic_kernel.functions.kernel_json_schema import KernelJsonSchema
from semantic_kernel.sk_pydantic import SKBaseModel


class KernelParameterMetadata(SKBaseModel):
    """
    Defines a KernelParameterMetadata object.

    Attributes:
        name (str): The name of the parameter.
        description (str): The description of the parameter.
        default_value (Any): The default value of the parameter.
        parameter_type (Type): The type of the parameter.
        schema (KernelJsonSchema): The schema of the parameter.
        is_required (bool): Indicates whether the parameter is required.
    """

    name: str = Field(default="")
    description: Optional[str] = Field(default="")
    default_value: Optional[Any] = Field(default=None)
    parameter_type: Optional[Type] = Field(default=None)
    schema: Optional[KernelJsonSchema] = Field(default=None)
    is_required: Optional[bool] = Field(default=False)
