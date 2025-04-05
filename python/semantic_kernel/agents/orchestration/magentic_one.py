# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from typing import ClassVar

from autogen_core import AgentId, MessageContext, SingleThreadedAgentRuntime, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.agents.orchestration.group_chat import (
    GroupChatAgentContainer,
    GroupChatRequestMessage,
    GroupChatResetMessage,
    GroupChatResponseMessage,
)
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase
from semantic_kernel.agents.orchestration.prompts._magentic_one_prompts import (
    ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
)
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.kernel_pydantic import KernelBaseModel
from semantic_kernel.prompt_template.kernel_prompt_template import KernelPromptTemplate
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)


class TaskStartMessage(KernelBaseModel):
    """Message to start a task."""

    body: str


class ProgressLedgerItem(KernelBaseModel):
    """A progress ledger item."""

    reason: str
    answer: str | bool


class ProgressLedger(KernelBaseModel):
    """A progress ledger."""

    is_request_satisfied: ProgressLedgerItem
    is_in_loop: ProgressLedgerItem
    is_progress_being_made: ProgressLedgerItem
    next_speaker: ProgressLedgerItem
    instruction_or_question: ProgressLedgerItem


class MagenticOneManager(GroupChatAgentContainer):
    """Container for the Magentic One pattern."""

    participant_descriptions: dict[str, str] = Field(default_factory=dict)
    participant_topics: dict[str, str] = Field(default_factory=dict)

    task: str | None = None
    facts: ChatMessageContent | None = None
    plan: ChatMessageContent | None = None

    stall_count: int = 0
    round_count: int = 0
    max_stall_count: int = 3

    def __init__(self, service: ChatCompletionClientBase, **kwargs):
        """Initialize the Magentic One container.

        Args:
            service (ChatCompletionServiceClient): A chat completion service client.
            **kwargs: Additional keyword arguments.
        """
        agent = ChatCompletionAgent(
            name="MagenticOneManager",
            description="A manager agent for the Magentic One pattern.",
            service=service,
        )
        super().__init__(agent=agent, **kwargs)

    @message_handler
    async def _on_task_start_message(self, message: TaskStartMessage, ctx: MessageContext) -> None:
        # Start the outer loop by creating the initial task ledger.
        self.task = message.body
        await self._run_outer_loop()

    @message_handler
    async def _on_group_chat_message(self, message: GroupChatResponseMessage, ctx: MessageContext) -> None:
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
                    content=await prompt_template.render(Kernel(), KernelArguments(task=self.task)),
                )
            )
            response = await self.agent.get_response(messages=temp_chat_history.messages)
            self.facts = response.message
            temp_chat_history.add_message(self.facts)

        # 2. Generate an initial plan
        if not self.plan:
            prompt_template = KernelPromptTemplate(
                prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT)
            )
            temp_chat_history.add_message(
                ChatMessageContent(
                    role=AuthorRole.USER,
                    content=await prompt_template.render(Kernel(), KernelArguments(team=self.participant_descriptions)),
                )
            )
            response = await self.agent.get_response(messages=temp_chat_history.messages)
            self.plan = response.message

        # 3. Create a fresh task ledger with the facts and plan.
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT)
        )
        task_ledger = await prompt_template.render(
            Kernel(),
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
                content=task_ledger,
                name=self.__class__.__name__,
            )
        )
        await self.publish_message(
            GroupChatResponseMessage(
                body=self.chat_history.messages[-1],
            ),
            TopicId(self.shared_topic_type, self.id.key),
        )
        logger.debug(f"Initial task ledger:\n{task_ledger}")

        # 5. Start the inner loop.
        await self._run_inner_loop()

    async def _run_inner_loop(self) -> None:
        self.round_count += 1

        # 1. Update the progress ledger
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_PROGRESS_LEDGER_PROMPT)
        )
        progress_ledger_prompt = await prompt_template.render(
            Kernel(),
            KernelArguments(
                task=self.task,
                team=self.participant_descriptions,
                names=", ".join(self.participant_descriptions.keys()),
            ),
        )
        temp_chat_history = self.chat_history.model_copy(deep=True)
        temp_chat_history.add_message(ChatMessageContent(role=AuthorRole.USER, content=progress_ledger_prompt))

        response = await self.agent.get_response(
            messages=temp_chat_history.messages,
            arguments=KernelArguments(
                # TODO(@taochen): Double check how to make sure the service support json output.
                settings=PromptExecutionSettings(extension_data={"response_format": ProgressLedger}),
            ),
        )
        current_progress_ledger = ProgressLedger.model_validate_json(response.message.content)
        logger.debug(f"Current progress ledger:\n{current_progress_ledger.model_dump_json(indent=2)}")

        # 2. Process the progress ledger
        # Check for task completion
        if current_progress_ledger.is_request_satisfied.answer:
            logger.debug("Task completed.")
            await self._prepare_final_answer()
            return
        # Check for stalling or looping
        if not current_progress_ledger.is_progress_being_made.answer or current_progress_ledger.is_in_loop.answer:
            self.stall_count += 1
        else:
            self.stall_count = max(0, self.stall_count - 1)

        if self.stall_count > self.max_stall_count:
            logger.debug("Stalling detected. Resetting the task.")
            await self._update_task_ledger()
            await self._reset_for_outer_loop()
            await self._run_outer_loop()
            return

        next_step = current_progress_ledger.instruction_or_question.answer
        self.chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=next_step,
                name=self.__class__.__name__,
            )
        )
        await self.publish_message(
            GroupChatResponseMessage(
                body=self.chat_history.messages[-1],
            ),
            TopicId(self.shared_topic_type, self.id.key),
        )

        # 3. Request the next speaker to speak
        next_speaker = current_progress_ledger.next_speaker.answer
        if next_speaker not in self.participant_descriptions:
            raise ValueError(f"Unknown speaker: {next_speaker}")

        logger.debug(f"Magentic One manager selected agent: {next_speaker}")

        await self.publish_message(
            GroupChatRequestMessage(),
            TopicId(self.participant_topics[next_speaker], self.id.key),
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
                    Kernel(),
                    KernelArguments(task=self.task, old_facts=self.facts),
                ),
            )
        )
        self.facts = await self.agent.get_response(messages=temp_chat_history.messages)
        temp_chat_history.add_message(self.facts)

        # 2. Update the plan
        prompt_template = KernelPromptTemplate(
            prompt_template_config=PromptTemplateConfig(template=ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT)
        )
        temp_chat_history.add_message(
            ChatMessageContent(
                role=AuthorRole.USER,
                content=await prompt_template.render(
                    Kernel(),
                    KernelArguments(team=self.participant_descriptions),
                ),
            )
        )
        self.plan = await self.agent.get_response(messages=temp_chat_history.messages)

    async def _reset_for_outer_loop(self) -> None:
        """Reset the orchestrator's chat history and all participants' chat histories."""
        self.chat_history.clear()
        await asyncio.gather(*[
            self.send_message(
                GroupChatResetMessage(),
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
                    Kernel(),
                    KernelArguments(task=self.task),
                ),
            )
        )

        response = await self.agent.get_response(messages=temp_chat_history.messages)
        print(response.content)


class MagenticOneOrchestration(OrchestrationBase):
    """The Magentic One pattern orchestration."""

    manager_service: ChatCompletionClientBase | None = None
    manager: MagenticOneManager | None = None

    MANAGER_TYPE: ClassVar[str] = "magentic_one_manager_container"

    @override
    async def _start(self, task: str, runtime: SingleThreadedAgentRuntime) -> None:
        """Start the Magentic One pattern."""
        should_stop = True
        try:
            runtime.start()
        except Exception:
            should_stop = False
            logger.warning("Runtime is already started outside of the pattern.")

        await runtime.publish_message(
            TaskStartMessage(body=task),
            TopicId(self.shared_topic_type, "default"),
        )

        if should_stop:
            await runtime.stop_when_idle()

    @override
    async def _register_agents(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            GroupChatAgentContainer.register(
                runtime,
                self._get_container_type(agent),
                lambda agent=agent: GroupChatAgentContainer(
                    agent,
                    description=agent.description,
                    shared_topic_type=self.shared_topic_type,
                ),
            )
            for agent in self.agents
        ])
        await MagenticOneManager.register(
            runtime,
            self.MANAGER_TYPE,
            lambda: self._create_manager(),
        )

    @override
    async def _add_subscriptions(self, runtime: SingleThreadedAgentRuntime) -> None:
        """Add subscriptions."""
        subscriptions: list[TypeSubscription] = []
        for agent in self.agents:
            subscriptions.append(TypeSubscription(self.shared_topic_type, self._get_container_type(agent)))
            subscriptions.append(TypeSubscription(self._get_container_topic(agent), self._get_container_type(agent)))
        await asyncio.gather(*[runtime.add_subscription(sub) for sub in subscriptions])
        await runtime.add_subscription(TypeSubscription(self.shared_topic_type, self.MANAGER_TYPE))

    def _create_manager(self) -> MagenticOneManager:
        """Create the manager."""
        if self.manager:
            return self.manager
        if not self.manager_service:
            raise ValueError("The manager service is required.")
        return MagenticOneManager(
            service=self.manager_service,
            participant_descriptions={agent.name: agent.description for agent in self.agents},
            participant_topics={agent.name: self._get_container_topic(agent) for agent in self.agents},
            shared_topic_type=self.shared_topic_type,
        )

    def _get_container_type(self, agent: Agent) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_magentic_one_container"

    def _get_container_topic(self, agent: Agent) -> str:
        """Get the container topic type for an agent."""
        return f"{agent.name}group_chat_topic_{self.shared_topic_type}"
