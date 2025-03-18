# Copyright (c) Microsoft. All rights reserved.


import asyncio
from typing import override

from autogen_core import AgentId, DefaultTopicId, MessageContext, message_handler
from pydantic import Field

from semantic_kernel.agents.patterns.group_chat.agents import BaseGroupChatAgent
from semantic_kernel.agents.patterns.group_chat.message_types import GroupChatMessage, GroupChatReset, RequestToSpeak
from semantic_kernel.agents.patterns.magentic_one._prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
)
from semantic_kernel.agents.patterns.magentic_one.message_types import TaskStartMessage
from semantic_kernel.agents.patterns.magentic_one.progress_ledger import ProgressLedger
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig


class Orchestrator(BaseGroupChatAgent):
    """Orchestrator agent for the Magentic One pattern."""

    participant_descriptions: dict[str, str] = Field(default_factory=dict)
    participant_topics: dict[str, str] = Field(default_factory=dict)

    task: str | None = None
    facts: ChatMessageContent | None = None
    plan: ChatMessageContent | None = None

    stall_count: int = 0
    round_count: int = 0
    max_stall_count: int = 3

    def __init__(self, participant_descriptions: dict[str, str], participant_topics: dict[str, str]):
        """Initialize the orchestrator agent."""
        super().__init__(
            name="MagenticOneOrchestrator",
            description="Orchestrator agent for the Magentic One pattern.",
            service=OpenAIChatCompletion(),
        )

        self.participant_descriptions = participant_descriptions
        self.participant_topics = participant_topics

    @message_handler
    async def _on_task_start_message(self, message: TaskStartMessage, ctx: MessageContext) -> None:
        # Start the outer loop by creating the initial task ledger.
        self.task = message.body
        await self._run_outer_loop()

    @override
    @message_handler
    async def _on_group_chat_message(self, message: GroupChatMessage, ctx: MessageContext) -> None:
        await super()._on_group_chat_message(message, ctx)
        await self._run_inner_loop()

    async def _run_outer_loop(self) -> None:
        # 1. Gather some initial facts
        temp_chat_history = ChatHistory()
        if not self.facts:
            prompt_template = KernelPromptTemplate(
                prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT)
            )
            temp_chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=await prompt_template.render(self.kernel, KernelArguments(task=self.task)),
                )
            )
            self.facts = await self.get_response(temp_chat_history)
            temp_chat_history.add_message(self.facts)

        # 2. Generate an initial plan
        if not self.plan:
            prompt_template = KernelPromptTemplate(
                prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT)
            )
            temp_chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=await prompt_template.render(
                        self.kernel, KernelArguments(team=self.participant_descriptions)
                    ),
                )
            )
            self.plan = await self.get_response(temp_chat_history)

        # 3. Create a fresh task ledger with the facts and plan.
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT)
        )
        task_ledger = await prompt_template.render(
            self.kernel,
            KernelArguments(
                task=self.task,
                team=self.participant_descriptions,
                facts=self.facts,
                plan=self.plan,
            ),
        )

        # 4. Publish the task ledger to the group chat.
        # Need to add the task ledger to the orchestrator's chat history
        # since the publisher won't receive the message it sends even though
        # the publisher also subscribes to the topic.
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                name=self.name,
                content=task_ledger,
            )
        )
        await self.publish_message(
            GroupChatMessage(
                body=self.chat_history.messages[-1],
            ),
            DefaultTopicId(type="GroupChatTopic"),
        )

        # 5. Start the inner loop.
        await self._run_inner_loop()

    async def _run_inner_loop(self) -> None:
        self.round_count += 1

        # 1. Update the progress ledger
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_PROGRESS_LEDGER_PROMPT)
        )
        progress_ledger_prompt = await prompt_template.render(
            self.kernel,
            KernelArguments(
                task=self.task,
                team=self.participant_descriptions,
                names=", ".join(self.participant_descriptions.keys()),
            ),
        )
        temp_chat_history = self.chat_history.model_copy(deep=True)
        temp_chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                name=self.name,
                content=progress_ledger_prompt,
            )
        )

        response = await self.get_response(
            temp_chat_history,
            arguments=KernelArguments(
                settings=OpenAIChatPromptExecutionSettings(response_format=ProgressLedger),
            ),
        )
        current_progress_ledger = ProgressLedger.model_validate_json(response.content)

        # 2. Process the progress ledger
        # Check for task completion
        if current_progress_ledger.is_request_satisfied.answer:
            await self._prepare_final_answer()
            return
        # Check for stalling or looping
        if not current_progress_ledger.is_progress_being_made.answer or current_progress_ledger.is_in_loop.answer:
            self.stall_count += 1
        else:
            self.stall_count = max(0, self.stall_count - 1)

        if self.stall_count > self.max_stall_count:
            await self._update_task_ledger()
            await self._reset_for_outer_loop()
            await self._run_outer_loop()
            return

        next_step = current_progress_ledger.instruction_or_question.answer
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                name=self.name,
                content=next_step,
            )
        )
        await self.publish_message(
            GroupChatMessage(
                body=self.chat_history.messages[-1],
            ),
            DefaultTopicId(type="GroupChatTopic"),
        )

        # 3. Request the next speaker to speak
        next_speaker = current_progress_ledger.next_speaker.answer
        if next_speaker not in self.participant_descriptions:
            raise ValueError(f"Unknown speaker: {next_speaker}")

        await self.publish_message(
            RequestToSpeak(),
            DefaultTopicId(type=self.participant_topics[next_speaker]),
        )

    async def _update_task_ledger(self) -> None:
        temp_chat_history = self.chat_history.model_copy(deep=True)

        # 1. Update the facts
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT)
        )
        temp_chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    self.kernel,
                    KernelArguments(task=self.task, old_facts=self.facts),
                ),
            )
        )
        self.facts = await self.get_response(temp_chat_history)
        temp_chat_history.add_message(self.facts)

        # 2. Update the plan
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT)
        )
        temp_chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    self.kernel,
                    KernelArguments(team=self.participant_descriptions),
                ),
            )
        )
        self.plan = await self.get_response(temp_chat_history)

    async def _reset_for_outer_loop(self) -> None:
        """Reset the orchestrator's chat history and all participants' chat histories."""
        self.chat_history.clear()
        await asyncio.gather(*[
            self.send_message(
                GroupChatReset(),
                recipient=AgentId(agent, "default"),
            )
            for agent in self.participant_topics
        ])

        self.stall_count = 0

    async def _prepare_final_answer(self) -> None:
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_FINAL_ANSWER_PROMPT)
        )
        temp_chat_history = self.chat_history.model_copy(deep=True)
        temp_chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    self.kernel,
                    KernelArguments(task=self.task),
                ),
            )
        )

        response = await self.get_response(temp_chat_history)
        print(response.content)
