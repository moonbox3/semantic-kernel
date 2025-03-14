# Copyright (c) Microsoft. All rights reserved.

from typing import Annotated, ClassVar, override
from urllib.request import urlopen

from autogen_core import DefaultTopicId, MessageContext, RoutedAgent, message_handler
from PIL import Image
from pydantic import Field

from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.patterns.group_chat.message_types import GroupChatMessage, RequestToSpeak
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_text_to_image import OpenAITextToImage
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.contents.image_content import ImageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig


class BaseGroupChatAgent(ChatCompletionAgent):
    chat_history: ChatHistory = Field(default_factory=ChatHistory)

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        self.chat_history.add_message(message.body)

    @message_handler
    async def _on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        response = await self.get_response(self.chat_history)
        print("############################################")
        print(f"{self.name}:\n{response.content}")
        print("############################################")
        await self.publish_message(GroupChatMessage(body=response), DefaultTopicId(type="GroupChatTopic"))

    @staticmethod
    def description() -> str:
        raise NotImplementedError


class WriterAgent(BaseGroupChatAgent):
    def __init__(self):
        super().__init__(
            name="WriterAgent",
            description="Writer for creating any text content.",
            instructions="You are a Writer. You produce good work.",
            service=OpenAIChatCompletion(),
        )

    @staticmethod
    @override
    def description() -> str:
        return "Writer for creating any text content."


class EditorAgent(BaseGroupChatAgent):
    def __init__(self):
        super().__init__(
            name="EditorAgent",
            description="Editor for planning and reviewing the content.",
            instructions=(
                "You are an Editor. Plan and guide the task given by the user. "
                "Provide critical feedbacks to the draft and illustration produced by Writer and Illustrator. "
                "Approve if the task is completed and the draft and illustration meets user's requirements."
            ),
            service=OpenAIChatCompletion(),
        )

    @staticmethod
    @override
    def description() -> str:
        return "Editor for planning and reviewing the content."


class ImageGeneratorPlugin(KernelBaseModel):
    image_generator: OpenAITextToImage = Field(default_factory=OpenAITextToImage)

    @kernel_function
    async def generate_image(
        self, description: Annotated[str, "The description of the image to generate."]
    ) -> ImageContent:
        image = await self.image_generator.generate_image(description, 1024, 1024)
        return ImageContent(uri=image)


class IllustratorAgent(BaseGroupChatAgent):
    def __init__(self):
        super().__init__(
            name="IllustratorAgent",
            description="Illustrator for creating photo-realistic illustrations.",
            instructions=(
                "You are an Illustrator. You use the generate_image tool to create images given user's requirement. "
                "Make sure the images have consistent characters and style."
            ),
            plugins=[ImageGeneratorPlugin()],
            service=OpenAIChatCompletion(),
        )

    @staticmethod
    @override
    def description() -> str:
        return "Illustrator for creating photo-realistic illustrations."

    @override
    @message_handler
    async def _on_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        response: ChatMessageContent = await self.get_response(self.chat_history)
        for chat_history_message in self.chat_history:
            for item in chat_history_message.items:
                if isinstance(item, FunctionResultContent) and isinstance(item.result, ImageContent):
                    img = Image.open(
                        urlopen(item.result.uri if isinstance(item.result.uri, str) else str(item.result.uri))
                    )
                    img.show()

        await self.publish_message(GroupChatMessage(body=response), DefaultTopicId(type="GroupChatTopic"))


class UserAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("User for providing final approval.")

    @message_handler
    async def handle_request_to_speak(self, message: RequestToSpeak, ctx: MessageContext) -> None:
        user_input = input("Enter your message, type 'APPROVE' to conclude the task: ")
        await self.publish_message(
            GroupChatMessage(body=ChatMessageContent(AuthorRole.USER, content=user_input)),
            DefaultTopicId(type="GroupChatTopic"),
        )

    @staticmethod
    @override
    def description() -> str:
        return "User for providing final approval."


class GroupChatManager(BaseGroupChatAgent):
    SELECTOR_PROMPT: ClassVar[str] = """
You are in a role play game. The following roles are available:
{{$roles}}.
Read the following conversation. Then select the next role from {{$participants}} to play. Only return the role.

{{$history}}

Read the above conversation. Then select the next role from {{$participants}} to play. Only return the name of the role.
"""

    participant_descriptions: dict[str, str] = Field(default_factory=dict)
    participant_topics: dict[str, str] = Field(default_factory=dict)

    def __init__(self, participant_descriptions: dict[str, str], participant_topics: dict[str, str]):
        super().__init__(
            name="GroupChatManager",
            description="Manager for the group chat.",
            prompt_template_config=PromptTemplateConfig(template=self.SELECTOR_PROMPT),
            service=OpenAIChatCompletion(),
        )

        self.participant_descriptions = participant_descriptions
        self.participant_topics = participant_topics

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        await super()._on_group_chat_message(message, ctx)

        if message.body.role == AuthorRole.USER and message.body.content == "APPROVE":
            return

        next_agent = await self._select_next_agent()
        print(f"Next agent: {next_agent}")
        await self.publish_message(RequestToSpeak(), DefaultTopicId(type=self.participant_topics[next_agent]))

    async def _select_next_agent(self):
        user_message = ChatMessageContent(AuthorRole.USER, content="Please select the next role.")
        response = await self.get_response(
            ChatHistory(messages=[user_message]),
            KernelArguments(
                roles="\n".join(
                    f"{role}: {description}" for role, description in self.participant_descriptions.items()
                ),
                participants=", ".join(self.participant_descriptions.keys()),
                history=self.chat_history,
            ),
        )

        for participant_name in self.participant_descriptions:
            if participant_name.lower() in response.content.lower():
                return participant_name

        raise ValueError(f"The selected participant is not found in the response: {response.content}.")
