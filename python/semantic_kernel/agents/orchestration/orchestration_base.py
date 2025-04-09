# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Union

from autogen_core import AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
from pydantic import Field

from semantic_kernel.agents.agent import Agent
from semantic_kernel.kernel_pydantic import KernelBaseModel

logger: logging.Logger = logging.getLogger(__name__)


class OrchestrationStartMessage(KernelBaseModel):
    """A orchestration start message type that kicks off the multi-agent orchestration."""

    pass


class OrchestrationResultMessage(KernelBaseModel):
    """A orchestration result message type that contains the result of the multi-agent orchestration."""

    body: Any


class OrchestrationAgent(RoutedAgent):
    """An agent that is part of the orchestration that is responsible for publishing the result to external topics."""

    def __init__(
        self,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        result_trigger_func: Callable[[Any], None] | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            internal_topic_type (str): The unique and internal topic type of this orchestration instance.
            external_topic_type (str | None): The unique and external topic type of this orchestration instance.
            result_trigger_func (Callable[[Any], Awaitable[None]] | None): A function that is called when the result is
                                                                           available.
        """
        self._internal_topic_type = internal_topic_type
        self._external_topic_type = external_topic_type
        self._result_trigger_func = result_trigger_func
        super().__init__(description="Orchestration Agent")

    @message_handler
    async def _handle_orchestration_result_message(
        self, message: OrchestrationResultMessage, ctx: MessageContext
    ) -> None:
        """Handle the orchestration result message."""
        # Simply route the message to the external topic
        if self._external_topic_type:
            await self.publish_message(message, TopicId(self._external_topic_type, self.id.key))

        # Call the result trigger function if provided
        if self._result_trigger_func:
            self._result_trigger_func(message.body)


class OrchestrationBase(KernelBaseModel, ABC):
    """Base class for multi-agent orchestration."""

    name: str = Field(
        default=__name__,
        description="The name of this orchestration instance.",
    )
    description: str = Field(
        default="A multi-agent orchestration instance.",
        description="The description of this orchestration instance.",
    )

    agents: list[Union[Agent, "OrchestrationBase"]] = Field(default_factory=list)

    internal_topic_type: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="The unique and internal topic type of this orchestration instance.",
    )

    async def invoke(
        self,
        task: str,
        runtime: AgentRuntime,
        time_out: int | None = None,
    ) -> Any:
        """Invoke the multi-agent orchestration and return the result.

        This is different from the `start` method, where clients need to wait for
        the orchestration result by subscribing to the external topic.

        This method is a blocking call that waits for the orchestration to finish
        and returns the result directly.

        Args:
            task (str): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
            time_out (int | None): The timeout (seconds) for the orchestration. If None, wait indefinitely.
        """
        orchestration_result: Any = None
        orchestration_result_event = asyncio.Event()

        def result_trigger_func(result: Any) -> None:
            nonlocal orchestration_result
            orchestration_result = result
            orchestration_result_event.set()

        # This unique ID is used to isolate the orchestration run from others.
        unique_registration_id = uuid.uuid4().hex

        await self.register(runtime, unique_registration_id, result_trigger_func=result_trigger_func)
        await self._start(task, runtime)

        # Wait for the orchestration result
        if time_out is not None:
            await asyncio.wait_for(orchestration_result_event.wait(), timeout=time_out)
        else:
            await orchestration_result_event.wait()

        if orchestration_result is None:
            raise RuntimeError("Orchestration result is None.")
        return orchestration_result

    async def start(self, task: str, runtime: AgentRuntime) -> None:
        """Start the multi-agent orchestration.

        This is different from the `invoke` method, where the result is returned directly.

        This method returns immediately and clients need to wait for the orchestration
        result by subscribing to the external topic.
        """
        # This unique ID is used to isolate the orchestration run from others.
        unique_registration_id = uuid.uuid4().hex

        await self.register(runtime, unique_registration_id)
        await self._start(task, runtime)

    async def register(
        self,
        runtime: AgentRuntime,
        unique_registration_id: str,
        external_topic_type: str | None = None,
        result_trigger_func: Callable[[Any], None] | None = None,
    ) -> None:
        """Registers the orchestration instance with the runtime."""
        await OrchestrationAgent.register(
            runtime,
            self._get_orchestration_agent_type(unique_registration_id),
            lambda: OrchestrationAgent(
                internal_topic_type=self.internal_topic_type,
                external_topic_type=external_topic_type,
                result_trigger_func=result_trigger_func,
            ),
        )
        await runtime.add_subscription(
            TypeSubscription(
                self.internal_topic_type,
                self._get_orchestration_agent_type(unique_registration_id),
            )
        )
        await runtime.add_subscription(
            TypeSubscription(
                external_topic_type,
                self._get_orchestration_agent_type(unique_registration_id),
            )
        )

        await self._register_agents(runtime, unique_registration_id)
        await self._add_subscriptions(runtime, unique_registration_id)

    @abstractmethod
    async def _start(self, task: str, runtime: AgentRuntime) -> None:
        """Start the multi-agent orchestration."""
        pass

    @abstractmethod
    async def _register_agents(self, runtime: AgentRuntime, unique_registration_id: str) -> None:
        """Register the agents."""
        pass

    @abstractmethod
    async def _add_subscriptions(self, runtime: AgentRuntime, unique_registration_id: str) -> None:
        """Add subscriptions."""
        pass

    def _get_orchestration_agent_type(self, unique_registration_id: str) -> str:
        """Get the collection agent type."""
        return f"{OrchestrationAgent.__name__}_{unique_registration_id}"
