# Copyright (c) Microsoft. All rights reserved.

from pydantic import Field

from semantic_kernel.connectors.ai.ai_request_settings import AIRequestSettings
from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.semantic_functions.prompt_template import PromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import PromptTemplateConfig


class KernelFunctionFromPrompt(KernelFunction):
    """
    A KernelFunction that is created from a PromptTemplate and a PromptTemplateConfig.

    Attributes:

        prompt_template (PromptTemplate): The PromptTemplate used to create the function.
        prompt_template_config (PromptTemplateConfig): The PromptTemplateConfig used to create the function.

    """

    prompt_template: PromptTemplate = Field(default_factory=PromptTemplate)
    prompt_template_config: PromptTemplateConfig = Field(default_factory=PromptTemplateConfig)

    def __init__(
        self,
        name: str,
        description: str,
        prompt_template_config: PromptTemplateConfig,
        execution_settings: AIRequestSettings,
    ):
        self.prompt_template_config = prompt_template_config
        kernel_parameter_metadata = prompt_template_config.get_kernel_parameter_metadata()
        kernel_return_parameter_metadata = prompt_template_config.get_kernel_return_parameter_metadata()

        super().__init__(
            name=name,
            description=description,
            parameters=kernel_parameter_metadata,
            return_parameters=kernel_return_parameter_metadata,
            execution_settings=execution_settings,
        )

    def create(
        prompt_template: PromptTemplate, prompt_template_config: PromptTemplateConfig, execution_settings
    ) -> KernelFunction:
        return KernelFunctionFromPrompt(
            name=prompt_template.name,
            description=prompt_template_config.description,
            prompt_template_config=prompt_template_config,
            execution_settings=execution_settings,
        )
