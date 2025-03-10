# Copyright (c) Microsoft. All rights reserved.

import logging
from queue import Queue
from typing import Any

from autogen_core import BaseAgent, MessageContext

from semantic_kernel.processes.autogen_runtime.messages import DequeueAllExternalEvents, EnqueueExternalEvent

logger = logging.getLogger(__name__)


class ExternalEventBufferAgent(BaseAgent):
    """Buffer agent for external events."""

    def __init__(self, agent_id: str):
        """Initialize the ExternalEventBufferAgent."""
        super().__init__(agent_id)
        self.queue: Queue[str] = Queue()

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """On message implementation to handle incoming messages."""
        if isinstance(message, EnqueueExternalEvent):
            self.queue.put(message.event_json)
            logger.info(f"[{self.id}] Enqueued external event: {message.event_json}")
            return None

        if isinstance(message, DequeueAllExternalEvents):
            items = []
            while not self.queue.empty():
                items.append(self.queue.get())
            logger.info(f"[{self.id}] Dequeued external events: {len(items)} items.")
            return items

        logger.warning(f"[{self.id}] Unhandled message type: {type(message)}")
        return None
