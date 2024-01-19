# Copyright (c) Microsoft. All rights reserved.

from typing import Dict, Any, List
from semantic_kernel.sk_pydantic import SKBaseModel
from abc import ABC, abstractmethod
import re
from pydantic import Field, field_validator

from semantic_kernel.functions.kernel_return_parameter_metadata import KernelReturnParameterMetadata
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.functions.kernel_function_metadata import KernelFunctionMetadata
from semantic_kernel.connectors.ai.ai_request_settings import AIRequestSettings


class KernelFunction(SKBaseModel, ABC):
    """
    Defines a KernelFunction object.

    Attributes:
        name (str): The name of the function.
        description (str): The description of the function.
        metadata (KernelFunctionMetadata): The metadata of the function.
        execution_settings (AIRequestSettings): The execution settings for the function.
    """
    name: str = Field(default="")
    description: str = Field(default="")
    metadata: KernelFunctionMetadata
    execution_settings: AIRequestSettings = Field(default_factory=AIRequestSettings) # TODO: support multiple execution settings

    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[KernelParameterMetadata],
        return_parameters: KernelReturnParameterMetadata,
        execution_settings: AIRequestSettings,
    ):
        """
        Initializes a new instance of the KernelFunction class.

        Args:
            name (str): The name of the function.
            description (str): The description of the function.
            parameters (List[KernelParameterMetadata]): The list of parameters for the function.
            return_parameters (KernelReturnParameterMetadata): The return parameter for the function.
            execution_settings (AIRequestSettings): The execution settings for the function.
        """
        metadata = KernelFunctionMetadata(
            name=name, description=description, parameters=parameters, return_parameter=return_parameters
        )
        super().__init__(name=name, description=description, metadata=metadata, execution_settings=execution_settings)

    async def invoke(self, kernel, kernel_arguments):
        # TODO: Implement this
        pass

    async def invoke_streaming(self, kernel, kernel_arguments):
        # TODO: Implement this
        pass