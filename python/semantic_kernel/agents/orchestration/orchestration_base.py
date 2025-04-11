# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar, Union, cast

from autogen_core import AgentRuntime, BaseAgent, MessageContext

from semantic_kernel.agents.agent import Agent

logger: logging.Logger = logging.getLogger(__name__)


TInternalInputMessage = TypeVar("TInternalInputMessage")
TInternalOutputMessage = TypeVar("TInternalOutputMessage")
TExternalInputMessage = TypeVar("TExternalInputMessage")
TExternalOutputMessage = TypeVar("TExternalOutputMessage")


class OrchestrationActorBase(
    BaseAgent,
    Generic[
        TExternalInputMessage,
        TInternalInputMessage,
        TInternalOutputMessage,
        TExternalOutputMessage,
    ],
):
    """An orchestrator actor that is part of the orchestration.

    This actor is responsible for relaying external messages to the internal topic or actor and vice versa.
    """

    def __init__(
        self,
        internal_topic_type: str,
        external_input_message_type: type[TExternalInputMessage],
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        input_transition: Callable[[TExternalInputMessage], Awaitable[TInternalInputMessage] | TInternalInputMessage]
        | None = None,
        output_transition: Callable[
            [TInternalOutputMessage], Awaitable[TExternalOutputMessage] | TExternalOutputMessage
        ]
        | None = None,
        result_callback: Callable[[TExternalOutputMessage], None] | None = None,
    ) -> None:
        """Initialize the orchestration agent.

        Args:
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            external_input_message_type (type[TExternalInputMessage]): The type of the external input message.
                This is for dynamic type checking.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            input_transition (Callable | None):
                A function that transforms the external input message to the internal input message.
            output_transition (Callable | None):
                A function that transforms the internal output message to the external output message.
            result_callback: A function that is called when the result is available.
        """
        self._internal_topic_type = internal_topic_type
        self._external_input_message_type = external_input_message_type

        self._external_topic_type = external_topic_type
        self._direct_actor_type = direct_actor_type
        self._result_callback = result_callback

        if input_transition is None:

            def input_transition_func(input_message: TExternalInputMessage) -> TInternalInputMessage:
                return cast(TInternalInputMessage, input_message)

            self._input_transition = input_transition_func
        else:
            self._input_transition = input_transition  # type: ignore[assignment]

        if output_transition is None:

            def output_transition_func(output_message: TInternalOutputMessage) -> TExternalOutputMessage:
                return cast(TExternalOutputMessage, output_message)

            self._output_transition = output_transition_func
        else:
            self._output_transition = output_transition  # type: ignore[assignment]

        super().__init__(description="Orchestration Agent")

    @abstractmethod
    async def _handle_orchestration_input_message(
        self,
        message: TExternalInputMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle the orchestration input message."""
        pass

    @abstractmethod
    async def _handle_orchestration_output_message(
        self,
        message: TInternalOutputMessage,
        ctx: MessageContext,
    ) -> None:
        """Handle the orchestration output message."""
        pass


class OrchestrationBase(
    ABC,
    Generic[
        TExternalInputMessage,
        TInternalInputMessage,
        TInternalOutputMessage,
        TExternalOutputMessage,
    ],
):
    """Base class for multi-agent orchestration."""

    def __init__(
        self,
        workers: list[Union[Agent, "OrchestrationBase"]],
        external_input_message_type: type[TExternalInputMessage],
        name: str | None = None,
        description: str | None = None,
        input_transition: Callable[[TExternalInputMessage], Awaitable[TInternalInputMessage] | TInternalInputMessage]
        | None = None,
        output_transition: Callable[
            [TInternalOutputMessage], Awaitable[TExternalOutputMessage] | TExternalOutputMessage
        ]
        | None = None,
    ) -> None:
        """Initialize the orchestration base.

        Args:
            workers (list[Union[Agent, OrchestrationBase]]): The list of agents or orchestrations to be used.
            external_input_message_type (type[TExternalInputMessage]): The type of the external input message.
                This is for dynamic type checking.
            name (str | None): A unique name of the orchestration. If None, a unique name will be generated.
            description (str | None): The description of the orchestration. If None, use a default description.
            input_transition (Callable | None):
                A function that transforms the external input message to the internal input message.
            output_transition (Callable | None):
                A function that transforms the internal output message to the external output message.
        """
        self.name = name or f"{self.__class__.__name__}_{uuid.uuid4().hex}"
        self.description = description or "A multi-agent orchestration."

        self._input_transition = input_transition
        self._output_transition = output_transition

        self._workers = workers
        self._external_input_message_type = external_input_message_type

    async def invoke(
        self,
        task: str,
        runtime: AgentRuntime,
        time_out: int | None = None,
    ) -> TExternalOutputMessage:
        """Invoke the multi-agent orchestration and return the result.

        This method is a blocking call that waits for the orchestration to finish
        and returns the result.

        Args:
            task (str): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
            time_out (int | None): The timeout (seconds) for the orchestration. If None, wait indefinitely.
        """
        orchestration_result: Any = None
        orchestration_result_event = asyncio.Event()

        def result_callback(result: Any) -> None:
            nonlocal orchestration_result
            orchestration_result = result
            orchestration_result_event.set()

        # This unique topic type is used to isolate the orchestration run from others.
        internal_topic_type = uuid.uuid4().hex

        await self._prepare(
            runtime,
            internal_topic_type=internal_topic_type,
            result_callback=result_callback,
        )
        await self._start(task, runtime, internal_topic_type)

        # Wait for the orchestration result
        if time_out is not None:
            await asyncio.wait_for(orchestration_result_event.wait(), timeout=time_out)
        else:
            await orchestration_result_event.wait()

        if orchestration_result is None:
            raise RuntimeError("Orchestration result is None.")
        return orchestration_result

    async def prepare(
        self,
        runtime: AgentRuntime,
        external_topic_type: str | None = None,
        direct_actor_type: str | None = None,
        result_callback: Callable[[TExternalOutputMessage], None] | None = None,
    ) -> str:
        """Prepare the orchestration with the runtime.

        Args:
            runtime (AgentRuntime): The runtime environment for the agents.
            external_topic_type (str | None): The external topic type for the orchestration to broadcast
                and receive messages. If set, the orchestration will subscribe itself to this topic.
            direct_actor_type (str | None): The direct actor type for which the orchestration
                actor will relay the output message to.
            result_callback (Callable[[TExternalOutputMessage], None] | None):
                A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        # This unique topic type is used to isolate the orchestration run from others.
        internal_topic_type = uuid.uuid4().hex

        return await self._prepare(
            runtime,
            internal_topic_type,
            external_topic_type=external_topic_type,
            direct_actor_type=direct_actor_type,
            result_callback=result_callback,
        )

    @abstractmethod
    async def _start(self, task: str, runtime: AgentRuntime, internal_topic_type: str) -> None:
        """Start the multi-agent orchestration.

        Args:
            task (str): The task to be executed by the agents.
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
        """
        pass

    @abstractmethod
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
            runtime (AgentRuntime): The runtime environment for the agents.
            internal_topic_type (str): The internal topic type for the orchestration that this actor is part of.
            external_topic_type (str | None): The external topic type for the orchestration.
            direct_actor_type (str | None): The direct actor type for which this actor will relay the output message to.
            result_callback (Callable[[TExternalOutputMessage], None] | None):
                A function that is called when the result is available.

        Returns:
            str: The actor type of the orchestration so that external actors can send messages to it.
        """
        pass
