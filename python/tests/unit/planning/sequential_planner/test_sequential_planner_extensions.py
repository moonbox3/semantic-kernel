# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import Mock, MagicMock

import pytest

from semantic_kernel.memory.memory_query_result import MemoryQueryResult
from semantic_kernel.memory.semantic_text_memory_base import SemanticTextMemoryBase
from semantic_kernel.orchestration.context_variables import ContextVariables
from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase
from semantic_kernel.orchestration.sk_function import SKFunction
from semantic_kernel.plugin_definition.default_kernel_plugin import DefaultKernelPlugin
from semantic_kernel.planning.sequential_planner.sequential_planner_config import (
    SequentialPlannerConfig,
)
from semantic_kernel.planning.sequential_planner.sequential_planner_extensions import (
    SequentialPlannerFunctionViewExtension,
    SequentialPlannerSKContextExtension,
)
from semantic_kernel.plugin_definition.function_view import FunctionView
from semantic_kernel.plugin_definition.functions_view import FunctionsView
from semantic_kernel.plugin_definition.kernel_plugin_collection import (
    KernelPluginCollection,
)
from semantic_kernel.plugin_definition.parameter_view import ParameterView
from semantic_kernel.semantic_functions.prompt_template_config import PromptTemplateConfig
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.semantic_function_config import SemanticFunctionConfig
from semantic_kernel.template_engine.prompt_template_engine import PromptTemplateEngine

async def _async_generator(query_result):
    yield query_result


@pytest.mark.asyncio
async def test_can_call_get_available_functions_with_no_functions_async():
    variables = ContextVariables()
    plugins = KernelPluginCollection()

    memory = Mock(spec=SemanticTextMemoryBase)
    memory_query_result = MemoryQueryResult(
        is_reference=False,
        id="id",
        text="text",
        description="description",
        external_source_name="sourceName",
        additional_metadata="value",
        relevance=0.8,
        embedding=None,
    )

    async_enumerable = _async_generator(memory_query_result)
    memory.search_async.return_value = async_enumerable

    # Arrange GetAvailableFunctionsAsync parameters
    context = SKContext(variables, memory, plugins)
    config = SequentialPlannerConfig()
    semantic_query = "test"

    # Act
    result = await SequentialPlannerSKContextExtension.get_available_functions_async(context, config, semantic_query)

    # Assert
    assert result is not None
    memory.search_async.assert_not_called()


@pytest.mark.asyncio
async def test_can_call_get_available_functions_with_functions_async():
    variables = ContextVariables()

    function_mock = Mock(spec=SKFunctionBase)
    functions_view = FunctionsView()
    native_function_view = FunctionView(
        "functionName",
        "MockPlugin",
        "description",
        [ParameterView(name='input', description='Mock input description', default_value='default_input_value', type_='string', required=False)],
        is_semantic=False,
        is_asynchronous=True,
    )
    functions_view.add_function(native_function_view)

    expected_plugin_name = "test_plugin"
    expected_plugin_description = "A unit test plugin"

    def mock_function(input: str, context: "SKContext") -> None:
        pass

    mock_function.__sk_function__ = True
    mock_function.__sk_function_name__ = "functionName"
    mock_function.__sk_function_description__ = "description"
    mock_function.__sk_function_input_description__ = "Mock input description"
    mock_function.__sk_function_input_default_value__ = "default_input_value"

    mock_method = mock_function

    native_function = SKFunction.from_native_method(mock_method, "MockPlugin")

    plugins = KernelPluginCollection()
    plugins.add(DefaultKernelPlugin(name=expected_plugin_name, description=expected_plugin_description, functions=[native_function]))

    memory_query_result = MemoryQueryResult(
        is_reference=False,
        id=SequentialPlannerFunctionViewExtension.to_fully_qualified_name(native_function_view),
        text="text",
        description="description",
        external_source_name="sourceName",
        additional_metadata="value",
        relevance=0.8,
        embedding=None,
    )

    async_enumerable = _async_generator(memory_query_result)
    memory = Mock(spec=SemanticTextMemoryBase)
    memory.search_async.return_value = async_enumerable

    # Arrange GetAvailableFunctionsAsync parameters
    context = SKContext.model_construct(variables=variables, memory=memory, plugin_collection=plugins)
    config = SequentialPlannerConfig()
    semantic_query = "test"

    # Act
    result = await SequentialPlannerSKContextExtension.get_available_functions_async(context, config, semantic_query)

    # Assert
    assert result is not None
    assert len(result) == 1
    assert result[0] == native_function_view


@pytest.mark.asyncio
async def test_can_call_get_available_functions_with_functions_and_relevancy_async():
    # Arrange
    variables = ContextVariables()

    # Arrange FunctionView
    function_mock = Mock(spec=SKFunctionBase)
    functions_view = FunctionsView()
    function_view = FunctionView(
        "functionName",
        "pluginName",
        "description",
        [],
        is_semantic=True,
        is_asynchronous=False,
    )
    native_function_view = FunctionView(
        "nativeFunctionName",
        "pluginName",
        "description",
        [],
        is_semantic=False,
        is_asynchronous=False,
    )
    functions_view.add_function(function_view)
    functions_view.add_function(native_function_view)

    # Arrange Mock Memory and Result
    memory_query_result = MemoryQueryResult(
        is_reference=False,
        id=SequentialPlannerFunctionViewExtension.to_fully_qualified_name(function_view),
        text="text",
        description="description",
        external_source_name="sourceName",
        additional_metadata="value",
        relevance=0.8,
        embedding=None,
    )
    memory = Mock(spec=SemanticTextMemoryBase)
    memory.search_async.return_value = _async_generator(memory_query_result)

    prompt_config = PromptTemplateConfig.from_execution_settings(max_tokens=2000, temperature=0.7, top_p=0.8)
    prompt_template = ChatPromptTemplate("{{$user_input}}", PromptTemplateEngine(), prompt_config)
    function_config = SemanticFunctionConfig(prompt_config, prompt_template)

    expected_plugin_name = "test_plugin"
    expected_function_name = "mock_function"
    semantic_function = SKFunction.from_semantic_config(
        plugin_name=expected_plugin_name, function_name=expected_function_name, function_config=function_config
    )

    expected_plugin_description = "A unit test plugin"

    plugin = DefaultKernelPlugin(
        name=expected_plugin_name, description=expected_plugin_description, functions=[semantic_function]
    )

    plugins = KernelPluginCollection()
    plugins.add(plugin)

    # Arrange GetAvailableFunctionsAsync parameters
    context = SKContext.model_construct(
        variables=variables,
        memory=memory,
        plugin_collection=plugins,
    )
    config = SequentialPlannerConfig(relevancy_threshold=0.78)
    semantic_query = "test"

    # Act
    result = await SequentialPlannerSKContextExtension.get_available_functions_async(context, config, semantic_query)

    # Assert
    assert result is not None
    assert len(result) == 1
    assert result[0] == function_view


@pytest.mark.asyncio
async def test_can_call_get_available_functions_async_with_default_relevancy_async():
    # Arrange
    variables = ContextVariables()
    plugins = KernelPluginCollection()

    # Arrange Mock Memory and Result
    memory_query_result = MemoryQueryResult(
        is_reference=False,
        id="id",
        text="text",
        description="description",
        external_source_name="sourceName",
        additional_metadata="value",
        relevance=0.8,
        embedding=None,
    )
    async_enumerable = _async_generator(memory_query_result)
    memory = Mock(spec=SemanticTextMemoryBase)
    memory.search_async.return_value = async_enumerable

    # Arrange GetAvailableFunctionsAsync parameters
    context = SKContext.model_construct(variables=variables, memory=memory, plugin_collection=plugins)
    config = SequentialPlannerConfig(relevancy_threshold=0.78)
    semantic_query = "test"

    # Act
    result = await SequentialPlannerSKContextExtension.get_available_functions_async(context, config, semantic_query)

    # Assert
    assert result is not None
    memory.search_async.assert_called_once()
