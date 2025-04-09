# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys

from autogen_core import AgentId, AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.container_base import ContainerBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationBase, OrchestrationResultMessage
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

logger: logging.Logger = logging.getLogger(__name__)


class ConcurrentRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentResponseMessage(KernelBaseModel):
    """A response message type for concurrent agents."""

    body: ChatMessageContent


class ConcurrentAgentContainer(ContainerBase):
    """A agent container for concurrent agents that process tasks."""

    collection_agent_type: str

    @message_handler
    async def _handle_message(self, message: ConcurrentRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) started processing..."
        )

        response = await self.agent.get_response(messages=message.body)

        logger.debug(
            f"Concurrent container (Container ID: {self.id}; Agent name: {self.agent.name}) finished processing."
        )

        await self.send_message(
            ConcurrentResponseMessage(body=response.message),
            AgentId(
                type=self.collection_agent_type,
                key="default",
            ),
        )


class CollectionAgent(RoutedAgent):
    """A agent container for collection results from concurrent agents."""

    def __init__(self, description: str, expected_answer_count: int, orchestration_agent_type: str) -> None:
        """Initialize the collection agent container."""
        self._expected_answer_count = expected_answer_count
        self._orchestration_agent_type = orchestration_agent_type
        self._results: dict[str, ChatMessageContent] = {}

        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: ConcurrentResponseMessage, ctx: MessageContext) -> None:
        # TODO(@taochen): Make this thread-safe
        self._results[message.body.name] = message.body

        if len(self._results) == self._expected_answer_count:
            logger.debug(f"Collection container (Container ID: {self.id}) finished processing all responses.")
            await self.send_message(
                OrchestrationResultMessage(body=self._results),
                AgentId(
                    type=self._orchestration_agent_type,
                    key="default",
                ),
            )


class ConcurrentOrchestration(OrchestrationBase):
    """A concurrent multi-agent pattern orchestration."""

    @override
    async def _start(self, task: str, runtime: AgentRuntime) -> None:
        """Start the concurrent pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)
        await runtime.publish_message(
            ConcurrentRequestMessage(body=message),
            topic_id=TopicId(self.internal_topic_type, "default"),
        )

    @override
    async def _register_agents(self, runtime: AgentRuntime, unique_registration_id: str) -> None:
        """Register the agents."""
        await asyncio.gather(*[
            ConcurrentAgentContainer.register(
                runtime,
                self._get_agent_type(agent, unique_registration_id),
                lambda agent=agent: ConcurrentAgentContainer(
                    agent,
                    collection_agent_type=self._get_collection_agent_type(unique_registration_id),
                ),
            )
            for agent in self.agents
        ])

        await CollectionAgent.register(
            runtime,
            self._get_collection_agent_type(unique_registration_id),
            lambda: CollectionAgent(
                description="An internal agent that is responsible for collection results",
                expected_answer_count=len(self.agents),
                orchestration_agent_type=self._get_orchestration_agent_type(unique_registration_id),
            ),
        )

    @override
    async def _add_subscriptions(self, runtime: AgentRuntime, unique_registration_id: str) -> None:
        """Add subscriptions."""
        await asyncio.gather(*[
            runtime.add_subscription(
                TypeSubscription(
                    self.internal_topic_type,
                    self._get_agent_type(agent, unique_registration_id),
                )
            )
            for agent in self.agents
        ])

    def _get_agent_type(self, agent: Agent, unique_registration_id: str) -> str:
        """Get the container type for an agent."""
        return f"{agent.name}_{unique_registration_id}"

    def _get_collection_agent_type(self, unique_registration_id: str) -> str:
        """Get the collection agent type."""
        return f"{CollectionAgent.__name__}_{unique_registration_id}"
