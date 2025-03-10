# Copyright (c) Microsoft. All rights reserved.

import logging
from queue import Queue
from typing import Any

from autogen_core import BaseAgent, MessageContext

from semantic_kernel.processes.autogen_runtime.messages import DequeueAllMessages, EnqueueMessage

logger = logging.getLogger(__name__)


class MessageBufferAgent(BaseAgent):
    """Generic message buffer agent for storing JSON messages in a queue."""

    def __init__(self, agent_id: str):
        """Initialize the MessageBufferAgent."""
        super().__init__(agent_id)
        self.queue: Queue[str] = Queue()

    async def on_message_impl(self, message: Any, context: MessageContext) -> Any:
        """On message implementation to handle incoming messages."""
        if isinstance(message, EnqueueMessage):
            self.queue.put(message.message_json)
            return None

        if isinstance(message, DequeueAllMessages):
            items = []
            while not self.queue.empty():
                items.append(self.queue.get())
            return items

        logger.warning(f"[{self.id}] Unhandled message type: {type(message)}")
        return None
