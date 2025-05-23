# Copyright (c) Microsoft. All rights reserved.

import logging
import sys
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, ClassVar

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AsyncStreamingChatCompletions,
    ChatChoice,
    ChatCompletions,
    ChatCompletionsToolCall,
    ChatRequestMessage,
    JsonSchemaFormat,
    StreamingChatChoiceUpdate,
    StreamingChatCompletionsUpdate,
    StreamingChatResponseToolCallUpdate,
)
from pydantic import BaseModel

from semantic_kernel.connectors.ai.azure_ai_inference import AzureAIInferenceChatPromptExecutionSettings
from semantic_kernel.connectors.ai.azure_ai_inference.services.azure_ai_inference_base import (
    AzureAIInferenceBase,
    AzureAIInferenceClientType,
)
from semantic_kernel.connectors.ai.azure_ai_inference.services.azure_ai_inference_tracing import AzureAIInferenceTracing
from semantic_kernel.connectors.ai.azure_ai_inference.services.utils import MESSAGE_CONVERTERS
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.completion_usage import CompletionUsage
from semantic_kernel.connectors.ai.function_calling_utils import update_settings_from_function_call_configuration
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceType
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import CMC_ITEM_TYPES, ChatMessageContent
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.streaming_chat_message_content import STREAMING_CMC_ITEM_TYPES as STREAMING_ITEM_TYPES
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
from semantic_kernel.contents.streaming_text_content import StreamingTextContent
from semantic_kernel.contents.text_content import TextContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.contents.utils.finish_reason import FinishReason
from semantic_kernel.exceptions.service_exceptions import ServiceInvalidExecutionSettingsError
from semantic_kernel.schema.kernel_json_schema_builder import KernelJsonSchemaBuilder
from semantic_kernel.utils.feature_stage_decorator import experimental

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.function_call_choice_configuration import FunctionCallChoiceConfiguration
    from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)


@experimental
class AzureAIInferenceChatCompletion(ChatCompletionClientBase, AzureAIInferenceBase):
    """Azure AI Inference Chat Completion Service."""

    SUPPORTS_FUNCTION_CALLING: ClassVar[bool] = True

    def __init__(
        self,
        ai_model_id: str,
        api_key: str | None = None,
        endpoint: str | None = None,
        service_id: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
        client: ChatCompletionsClient | None = None,
        instruction_role: str | None = None,
    ) -> None:
        """Initialize the Azure AI Inference Chat Completion service.

        If no arguments are provided, the service will attempt to load the settings from the environment.
        The following environment variables are used:
        - AZURE_AI_INFERENCE_API_KEY
        - AZURE_AI_INFERENCE_ENDPOINT

        Args:
            ai_model_id: (str): A string that is used to identify the model such as the model name. (Required)
            api_key (str | None): The API key for the Azure AI Inference service deployment. (Optional)
            endpoint (str | None): The endpoint of the Azure AI Inference service deployment. (Optional)
            service_id (str | None): Service ID for the chat completion service. (Optional)
            env_file_path (str | None): The path to the environment file. (Optional)
            env_file_encoding (str | None): The encoding of the environment file. (Optional)
            client (ChatCompletionsClient | None): The Azure AI Inference client to use. (Optional)
            instruction_role (str | None): The role to use for 'instruction' messages, for example, summarization
                prompts could use `developer` or `system`. (Optional)

        Raises:
            ServiceInitializationError: If an error occurs during initialization.
        """
        args: dict[str, Any] = {
            "ai_model_id": ai_model_id,
            "api_key": api_key,
            "client_type": AzureAIInferenceClientType.ChatCompletions,
            "client": client,
            "endpoint": endpoint,
            "env_file_path": env_file_path,
            "env_file_encoding": env_file_encoding,
        }

        if service_id:
            args["service_id"] = service_id

        if instruction_role:
            args["instruction_role"] = instruction_role

        super().__init__(**args)

    # region Overriding base class methods

    # Override from AIServiceClientBase
    @override
    def get_prompt_execution_settings_class(self) -> type["PromptExecutionSettings"]:
        return AzureAIInferenceChatPromptExecutionSettings

    # Override from AIServiceClientBase
    @override
    def service_url(self) -> str | None:
        if hasattr(self.client, "_client") and hasattr(self.client._client, "_base_url"):
            # Best effort to get the endpoint
            return self.client._client._base_url
        return None

    @override
    async def _inner_get_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
    ) -> list["ChatMessageContent"]:
        if not isinstance(settings, AzureAIInferenceChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AzureAIInferenceChatPromptExecutionSettings)  # nosec

        assert isinstance(self.client, ChatCompletionsClient)  # nosec
        with AzureAIInferenceTracing():
            settings_dict = settings.prepare_settings_dict()
            self._handle_structured_output(settings, settings_dict)
            response: ChatCompletions = await self.client.complete(
                messages=self._prepare_chat_history_for_request(chat_history),
                # The model id will be ignored by the service if the endpoint serves only one model (i.e. MaaS)
                model=self.ai_model_id,
                model_extras=settings.extra_parameters,
                **settings_dict,
            )
        response_metadata = self._get_metadata_from_response(response)

        return [self._create_chat_message_content(response, choice, response_metadata) for choice in response.choices]

    @override
    async def _inner_get_streaming_chat_message_contents(
        self,
        chat_history: "ChatHistory",
        settings: "PromptExecutionSettings",
        function_invoke_attempt: int = 0,
    ) -> AsyncGenerator[list["StreamingChatMessageContent"], Any]:
        if not isinstance(settings, AzureAIInferenceChatPromptExecutionSettings):
            settings = self.get_prompt_execution_settings_from_settings(settings)
        assert isinstance(settings, AzureAIInferenceChatPromptExecutionSettings)  # nosec

        assert isinstance(self.client, ChatCompletionsClient)  # nosec
        with AzureAIInferenceTracing():
            settings_dict = settings.prepare_settings_dict()
            self._handle_structured_output(settings, settings_dict)
            response: AsyncStreamingChatCompletions = await self.client.complete(
                stream=True,
                # The model id will be ignored by the service if the endpoint serves only one model (i.e. MaaS)
                model=self.ai_model_id,
                messages=self._prepare_chat_history_for_request(chat_history),
                model_extras=settings.extra_parameters,
                **settings_dict,
            )

        async for chunk in response:
            if len(chunk.choices) == 0:
                continue
            chunk_metadata = self._get_metadata_from_response(chunk)
            yield [
                self._create_streaming_chat_message_content(chunk, choice, chunk_metadata, function_invoke_attempt)
                for choice in chunk.choices
            ]

    @override
    def _verify_function_choice_settings(self, settings: "PromptExecutionSettings") -> None:
        if not isinstance(settings, AzureAIInferenceChatPromptExecutionSettings):
            raise ServiceInvalidExecutionSettingsError(
                "The settings must be an AzureAIInferenceChatPromptExecutionSettings."
            )
        if settings.extra_parameters is not None and settings.extra_parameters.get("n", 1) > 1:
            # Currently only OpenAI models allow multiple completions but the Azure AI Inference service
            # does not expose the functionality directly. If users want to have more than 1 responses, they
            # need to configure `extra_parameters` with a key of "n" and a value greater than 1.
            raise ServiceInvalidExecutionSettingsError(
                "Auto invocation of tool calls may only be used with a single completion."
            )

    @override
    def _update_function_choice_settings_callback(
        self,
    ) -> Callable[["FunctionCallChoiceConfiguration", "PromptExecutionSettings", FunctionChoiceType], None]:
        return update_settings_from_function_call_configuration

    @override
    def _reset_function_choice_settings(self, settings: "PromptExecutionSettings") -> None:
        if hasattr(settings, "tool_choice"):
            settings.tool_choice = None
        if hasattr(settings, "tools"):
            settings.tools = None

    @override
    def _prepare_chat_history_for_request(
        self,
        chat_history: ChatHistory,
        role_key: str = "role",
        content_key: str = "content",
    ) -> list[ChatRequestMessage]:
        chat_request_messages: list[ChatRequestMessage] = []

        for message in chat_history.messages:
            # If instruction_role is 'developer' and the message role is 'system', change it to 'developer'
            role = (
                AuthorRole.DEVELOPER
                if self.instruction_role == "developer" and message.role == AuthorRole.SYSTEM
                else message.role
            )
            chat_request_messages.append(MESSAGE_CONVERTERS[role](message))

        return chat_request_messages

    def _handle_structured_output(
        self, request_settings: AzureAIInferenceChatPromptExecutionSettings, settings: dict[str, Any]
    ) -> None:
        response_format = getattr(request_settings, "response_format", None)
        if getattr(request_settings, "structured_json_response", False) and response_format:
            # Case 1: response_format is a Pydantic BaseModel type
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                schema = response_format.model_json_schema()
                settings["response_format"] = JsonSchemaFormat(
                    name=response_format.__name__,
                    schema=schema,
                    description=f"Schema for {response_format.__name__}",
                    strict=True,
                )
            # Case 2: response_format is a type but not a subclass of BaseModel
            elif isinstance(response_format, type):
                generated_schema = KernelJsonSchemaBuilder.build(parameter_type=response_format, structured_output=True)
                assert generated_schema is not None  # nosec
                settings["response_format"] = JsonSchemaFormat(
                    name=response_format.__name__,
                    schema=generated_schema,
                    description=f"Schema for {response_format.__name__}",
                    strict=True,
                )
            # Case 3: response_format is already a JsonSchemaFormat instance, pass it
            elif isinstance(response_format, JsonSchemaFormat):
                settings["response_format"] = response_format
            # Case 4: response_format is a dictionary (legacy), create JsonSchemaFormat from dict
            elif isinstance(response_format, dict):
                settings["response_format"] = JsonSchemaFormat(**response_format)

    # endregion

    # region Non-streaming

    def _create_chat_message_content(
        self, response: ChatCompletions, choice: ChatChoice, metadata: dict[str, Any]
    ) -> ChatMessageContent:
        """Create a chat message content object.

        Args:
            response: The response from the service.
            choice: The choice from the response.
            metadata: The metadata from the response.

        Returns:
            A chat message content object.
        """
        items: list[CMC_ITEM_TYPES] = []
        if choice.message.content:
            items.append(
                TextContent(
                    text=choice.message.content,
                    metadata=metadata,
                )
            )
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                if isinstance(tool_call, ChatCompletionsToolCall):
                    items.append(
                        FunctionCallContent(
                            id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                    )

        return ChatMessageContent(
            role=AuthorRole(choice.message.role),
            items=items,
            inner_content=response,
            finish_reason=FinishReason(choice.finish_reason) if choice.finish_reason else None,
            metadata=metadata,
        )

    # endregion

    # region Streaming

    def _create_streaming_chat_message_content(
        self,
        chunk: AsyncStreamingChatCompletions,
        choice: StreamingChatChoiceUpdate,
        metadata: dict[str, Any],
        function_invoke_attempt: int,
    ) -> StreamingChatMessageContent:
        """Create a streaming chat message content object.

        Args:
            chunk: The chunk from the response.
            choice: The choice from the response.
            metadata: The metadata from the response.
            function_invoke_attempt: The function invoke attempt.

        Returns:
            A streaming chat message content object.
        """
        items: list[STREAMING_ITEM_TYPES] = []
        if choice.delta.content:
            items.append(
                StreamingTextContent(
                    choice_index=choice.index,
                    text=choice.delta.content,
                    inner_content=chunk,
                    metadata=metadata,
                )
            )
        if choice.delta.tool_calls:
            for tool_call in choice.delta.tool_calls:
                if isinstance(tool_call, StreamingChatResponseToolCallUpdate):
                    items.append(
                        FunctionCallContent(
                            id=tool_call.id,
                            index=choice.index,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        )
                    )

        return StreamingChatMessageContent(
            role=(AuthorRole(choice.delta.role) if choice.delta and choice.delta.role else AuthorRole.ASSISTANT),
            items=items,
            choice_index=choice.index,
            inner_content=chunk,
            finish_reason=FinishReason(choice.finish_reason) if choice.finish_reason else None,
            metadata=metadata,
            function_invoke_attempt=function_invoke_attempt,
            ai_model_id=self.ai_model_id,
        )

    # endregion

    def _get_metadata_from_response(self, response: ChatCompletions | StreamingChatCompletionsUpdate) -> dict[str, Any]:
        """Get metadata from the response.

        Args:
            response: The response from the service.

        Returns:
            A dictionary containing metadata.
        """
        return {
            "id": response.id,
            "model": response.model,
            "created": response.created,
            "usage": CompletionUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            if response.usage
            else None,
        }
