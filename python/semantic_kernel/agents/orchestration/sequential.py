# Copyright (c) Microsoft. All rights reserved.

import logging
import sys
from collections.abc import Awaitable, Callable
from typing import Any, Union

from autogen_core import AgentId, AgentRuntime, MessageContext, RoutedAgent, TopicId, TypeSubscription, message_handler
from typing_extensions import TypeVar

from semantic_kernel.agents.agent import Agent
from semantic_kernel.agents.orchestration.agent_actor_base import AgentActorBase
from semantic_kernel.agents.orchestration.orchestration_base import OrchestrationActorBase, OrchestrationBase
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel_pydantic import KernelBaseModel

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


logger: logging.Logger = logging.getLogger(__name__)


class SequentialRequestMessage(KernelBaseModel):
    """A request message type for concurrent agents."""

    body: ChatMessageContent


class SequentialResultMessage(KernelBaseModel):
    """A result message type for concurrent agents."""

    body: ChatMessageContent


TExternalInputMessage = TypeVar("TExternalInputMessage", default=SequentialRequestMessage)
TExternalOutputMessage = TypeVar("TExternalOutputMessage", default=SequentialResultMessage)


class SequentialOrchestrationActor(
    OrchestrationActorBase[
        TExternalInputMessage,
        SequentialRequestMessage,
        SequentialResultMessage,
        TExternalOutputMessage,
    ]
):
    """An agent that is part of the orchestration that is responsible for relaying external messages."""

    def __init__(
        self,
        internal_topic_type: str,
        external_input_message_type: type[TExternalInputMessage],
        initial_actor_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        input_transition: Callable[[TExternalInputMessage], Awaitable[SequentialRequestMessage]] | None = None,
        output_transition: Callable[[SequentialResultMessage], Awaitable[TExternalOutputMessage]] | None = None,
        result_callback: Callable[[TExternalOutputMessage], None] | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            external_input_message_type (type[TExternalInputMessage]): The type of the external input message.
                This is for dynamic type checking.
            initial_actor_type (str): The actor type of the first actor in the sequence.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            input_transition (Callable[[TExternalInputMessage], Awaitable[SequentialRequestMessage]] | None):
                A function that transforms the external input message to the internal input message.
            output_transition (Callable[[SequentialResultMessage], Awaitable[TExternalOutputMessage]] | None):
                A function that transforms the internal output message to the external output message.
            result_callback: A function that is called when the result is available.
        """
        self._initial_actor_type = initial_actor_type

        super().__init__(
            internal_topic_type=internal_topic_type,
            external_input_message_type=external_input_message_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            input_transition=input_transition,
            output_transition=output_transition,
            result_callback=result_callback,
        )

    @override
    async def on_message_impl(self, message: Any, ctx: MessageContext) -> None:
        if isinstance(message, SequentialRequestMessage):
            await self._handle_orchestration_input_message(message, ctx)
        elif isinstance(message, self._external_input_message_type):
            message: SequentialRequestMessage = await self._input_transition(message)
            await self._handle_orchestration_input_message(message, ctx)
        elif isinstance(message, SequentialResultMessage):
            await self._handle_orchestration_output_message(message, ctx)
        else:
            # Since the orchestration actor subscribes to the external topic type,
            # it may receive messages that are not of the expected type.
            pass

    @override
    async def _handle_orchestration_input_message(
        self,
        message: SequentialRequestMessage,
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration input message.")
        logger.debug(f"Relaying message to agent: {self._initial_actor_type}")
        await self.send_message(message, AgentId(type=self._initial_actor_type, key="default"))

    @override
    async def _handle_orchestration_output_message(
        self,
        message: SequentialResultMessage,
        ctx: MessageContext,
    ) -> None:
        logger.debug(f"{self.id}: Received orchestration output message.")
        external_output_message = await self._output_transition(message)

        if self._external_topic_type:
            logger.debug(f"Relaying message to external topic: {self._external_topic_type}")
            await self.publish_message(
                external_output_message,
                TopicId(self._external_topic_type, "default"),
            )
        if self._direct_actor_type:
            logger.debug(f"Relaying message directly to actor: {self._direct_actor_type}")
            await self.send_message(
                external_output_message,
                AgentId(self._direct_actor_type, "default"),
            )
        if self._result_callback:
            self._result_callback(external_output_message)


class SequentialAgentActor(AgentActorBase):
    """A agent actor for sequential agents that process tasks."""

    def __init__(self, agent: Agent, next_agent_type: str, **kwargs) -> None:
        """Initialize the agent actor."""
        self._next_agent_type = next_agent_type
        super().__init__(agent=agent, **kwargs)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        """Handle a message."""
        logger.debug(f"Sequential actor (Actor ID: {self.id}; Agent name: {self._agent.name}) started processing...")

        response = await self._agent.get_response(messages=message.body)

        logger.debug(f"Sequential actor (Actor ID: {self.id}; Agent name: {self._agent.name}) finished processing.")

        await self.send_message(
            SequentialRequestMessage(body=response.message),
            AgentId(
                type=self._next_agent_type,
                key="default",
            ),
        )


class CollectionActor(RoutedAgent):
    """A agent container for collection results from the last agent in the sequence."""

    def __init__(self, description: str, orchestration_agent_type: str) -> None:
        """Initialize the collection agent container."""
        self._orchestration_agent_type = orchestration_agent_type

        super().__init__(description=description)

    @message_handler
    async def _handle_message(self, message: SequentialRequestMessage, ctx: MessageContext) -> None:
        await self.send_message(
            SequentialResultMessage(body=message.body),
            AgentId(
                type=self._orchestration_agent_type,
                key="default",
            ),
        )


class SequentialOrchestration(
    OrchestrationBase[
        TExternalInputMessage,
        SequentialRequestMessage,
        SequentialResultMessage,
        TExternalOutputMessage,
    ]
):
    """A sequential multi-agent pattern orchestration."""

    def __init__(
        self,
        workers: list[Union[Agent, "OrchestrationBase"]],
        external_input_message_type: type[TExternalInputMessage] = SequentialRequestMessage,
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalInputMessage], Awaitable[SequentialRequestMessage]] | None = None,
        output_transition: Callable[[SequentialResultMessage], Awaitable[TExternalOutputMessage]] | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            workers (list[Union[Agent, OrchestrationBase]]): The list of agents or orchestrations to be used.
            external_input_message_type (type[TExternalInputMessage]): The type of the external input message.
                This is for dynamic type checking. Default is SequentialRequestMessage.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transition (Callable[[TExternalInputMessage], Awaitable[SequentialRequestMessage]] | None):
                A function that transforms the external input message to the internal input message.
            output_transition (Callable[[SequentialResultMessage], Awaitable[TExternalOutputMessage]] | None):
                A function that transforms the internal output message to the external output message.
        """
        super().__init__(
            workers,
            external_input_message_type=external_input_message_type,
            name=name,
            description=description,
            input_transition=input_transition,
            output_transition=output_transition,
        )

    @override
    async def _start(self, task: str, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Start the sequential pattern."""
        message = ChatMessageContent(AuthorRole.USER, content=task)
        await runtime.send_message(
            SequentialRequestMessage(body=message),
            AgentId(
                type=self._get_orchestration_actor_type(internal_topic_type),
                key="default",
            ),
        )

    @override
    async def _prepare(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOutputMessage], None] | None = None,
    ) -> str:
        """Register the actors and orchestrations with the runtime and add the required subscriptions.

        Args:
            runtime (AgentRuntime): The agent runtime.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
                Since the sequential orchestration doesn't broadcast messages internally, this is only used to
                uniquely identify the orchestration.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            result_callback: A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        initial_actor_type = await self._register_workers(runtime, internal_topic_type)
        await self._register_orchestration_actor(
            runtime,
            internal_topic_type,
            initial_actor_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )
        await self._register_collection_actor(runtime, internal_topic_type)
        await self._add_subscriptions(runtime, internal_topic_type, external_topic_type)

        return self._get_orchestration_actor_type(internal_topic_type)

    async def _register_orchestration_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        initial_actor_type: str,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOutputMessage], None] | None = None,
    ) -> None:
        """Register the orchestration actor."""
        await SequentialOrchestrationActor.register(
            runtime,
            self._get_orchestration_actor_type(internal_topic_type),
            lambda: SequentialOrchestrationActor(
                internal_topic_type,
                external_input_message_type=self._external_input_message_type,
                initial_actor_type=initial_actor_type,
                external_topic_type=external_topic_type,
                direct_actor_type=direct_actor_type,
                input_transition=self._input_transition,
                output_transition=self._output_transition,
                result_callback=result_callback,
            ),
        )

    async def _register_workers(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> str:
        """Register the workers.

        The workers will be registered in the reverse order so that the actor type of the next worker
        is available when the current worker is registered. This is important for the sequential
        orchestration, where actors need to know its next actor type to send the message to.

        Args:
            runtime (AgentRuntime): The agent runtime.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.

        Returns:
            str: The first actor type in the sequence.
        """
        next_actor_type = self._get_collection_actor_type(internal_topic_type)
        for index, worker in enumerate(reversed(self._workers)):
            if isinstance(worker, Agent):
                await SequentialAgentActor.register(
                    runtime,
                    self._get_agent_actor_type(worker, internal_topic_type),
                    lambda worker=worker, next_actor_type=next_actor_type: SequentialAgentActor(
                        worker,
                        next_agent_type=next_actor_type,
                    ),
                )
                logger.debug(
                    f"Registered agent actor of type {self._get_agent_actor_type(worker, internal_topic_type)}"
                )
                next_actor_type = self._get_agent_actor_type(worker, internal_topic_type)
            elif isinstance(worker, OrchestrationBase):
                worker_orchestration_actor_type = await worker.prepare(
                    runtime,
                    direct_actor_type=next_actor_type,
                )
                logger.debug(f"Registered orchestration actor of type {worker_orchestration_actor_type}")
                next_actor_type = worker_orchestration_actor_type
            else:
                raise TypeError(f"Unsupported node type: {type(worker)}")

        return next_actor_type

    async def _register_collection_actor(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
    ) -> None:
        """Register the collection actor."""
        await CollectionActor.register(
            runtime,
            self._get_collection_actor_type(internal_topic_type),
            lambda: CollectionActor(
                description="An internal agent that is responsible for collection results",
                orchestration_agent_type=self._get_orchestration_actor_type(internal_topic_type),
            ),
        )

    async def _add_subscriptions(
        self,
        runtime: AgentRuntime,
        internal_topic_type: str,
        external_topic_type: str | None = None,
    ) -> None:
        """Add subscriptions to the runtime."""
        if external_topic_type:
            await runtime.add_subscription(
                TypeSubscription(
                    external_topic_type,
                    self._get_orchestration_actor_type(internal_topic_type),
                )
            )

    def _get_agent_actor_type(self, agent: Agent, internal_topic_type: str) -> str:
        """Get the actor type for an agent.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{agent.name}_{internal_topic_type}"

    def _get_collection_actor_type(self, internal_topic_type: str) -> str:
        """Get the collection actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{CollectionActor.__name__}_{internal_topic_type}"

    def _get_orchestration_actor_type(self, internal_topic_type: str) -> str:
        """Get the orchestration actor type.

        The type is appended with the internal topic type to ensure uniqueness in the runtime
        that may be shared by multiple orchestrations.
        """
        return f"{SequentialOrchestrationActor.__name__}_{internal_topic_type}"
