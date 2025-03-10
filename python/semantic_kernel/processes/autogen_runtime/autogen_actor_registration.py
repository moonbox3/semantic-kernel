# Copyright (c) Microsoft. All rights reserved.

from typing import Callable

from autogen_core import SingleThreadedAgentRuntime

from semantic_kernel.processes.autogen_runtime.event_buffer_agent import EventBufferAgent
from semantic_kernel.processes.autogen_runtime.external_event_buffer_agent import ExternalEventBufferAgent
from semantic_kernel.processes.autogen_runtime.message_buffer_agent import MessageBufferAgent
from semantic_kernel.processes.autogen_runtime.process_agent import ProcessAgent
from semantic_kernel.processes.autogen_runtime.step_agent import StepAgent


async def register_autogen_agents(
    runtime: SingleThreadedAgentRuntime,
    factories: dict[str, Callable] | None = None,
) -> None:
    """Registers agent types with SingleThreadedAgentRuntime.

    The 'factories' is specifically for "step" logic references, if needed by StepAgent or ProcessAgent.
    """
    if factories is None:
        factories = {}

    async def process_agent_factory():  # noqa: RUF029
        return ProcessAgent("process_agent", factories, runtime)

    await runtime.register_factory("process_agent", agent_factory=process_agent_factory, expected_class=ProcessAgent)

    async def step_agent_factory():  # noqa: RUF029
        return StepAgent("step_agent", factories)

    await runtime.register_factory("step_agent", agent_factory=step_agent_factory, expected_class=StepAgent)

    async def message_buffer_factory():  # noqa: RUF029
        return MessageBufferAgent("MessageBufferAgent")

    await runtime.register_factory(
        "message_buffer_agent", agent_factory=message_buffer_factory, expected_class=MessageBufferAgent
    )

    async def event_buffer_factory():  # noqa: RUF029
        return EventBufferAgent("EventBufferAgent")

    await runtime.register_factory(
        "event_buffer_agent", agent_factory=event_buffer_factory, expected_class=EventBufferAgent
    )

    async def external_event_buffer_factory():  # noqa: RUF029
        return ExternalEventBufferAgent("ExternalEventBufferAgent")

    await runtime.register_factory(
        "external_event_buffer_agent",
        agent_factory=external_event_buffer_factory,
        expected_class=ExternalEventBufferAgent,
    )
