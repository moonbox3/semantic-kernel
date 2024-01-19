# Copyright (c) Microsoft. All rights reserved.

from typing import Any, ClassVar, Dict, List, Optional, Union
import re
from pydantic import Field, field_validator, root_validator
from semantic_kernel.sk_pydantic import SKBaseModel
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.functions.kernel_return_parameter_metadata import KernelReturnParameterMetadata


class KernelFunctionMetadata(SKBaseModel):
    """
    Defines a KernelFunctionMetadata object.

    Attributes:
        name (str): The name of the function.
        description (str): The description of the function.
        parameters (List[KernelParameterMetadata]): The list of parameters for the function.
        return_parameter (KernelReturnParameterMetadata): The return parameter for the function.
    """
    name: str = Field(default="")
    description: Optional[str] = Field(default="")
    parameters: Optional[List[KernelParameterMetadata]] = Field(default_factory=list)
    return_parameter: Optional[KernelReturnParameterMetadata] = Field(default=None)

    @field_validator("name", mode="after")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        """Validates that the name contains only uppercase, lowercase letters, or underscores."""
        pattern = r"^[A-Za-z_]+$"
        if not re.match(pattern, v):
            raise ValueError("Name must contain only uppercase, lowercase letters, or underscores")
        return v
