# Copyright (c) Microsoft. All rights reserved.

import logging
from queue import Queue
from typing import Any

from autogen_core import AgentId, BaseAgent, MessageContext

from semantic_kernel.processes.core_runtime.messages import (
    DequeueAllMessages,
    EnqueueMessage,
)

logger = logging.getLogger(__name__)


class EventBufferAgent(BaseAgent):
    """Buffer agent for step events. No @message_handler, just override on_message_impl()."""

    def __init__(self, agent_id: AgentId):
        """Initialize the EventBufferAgent."""
        super().__init__(agent_id.key)
        self.queue: Queue[str] = Queue()

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """On_message_impl method to handle incoming messages."""
        if isinstance(message, EnqueueMessage):
            self.queue.put(message.message_json)
            logger.info(f"[{self.id}] Enqueued message JSON: {message.message_json}")
            return None

        if isinstance(message, DequeueAllMessages):
            items = []
            while not self.queue.empty():
                items.append(self.queue.get())
            logger.info(f"[{self.id}] Dequeued all events: {len(items)} items.")
            return items

        logger.warning(f"[{self.id}] Unhandled message type: {type(message)}")
        return None
