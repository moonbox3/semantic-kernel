# Copyright (c) Microsoft. All rights reserved.


import logging
from typing import Any

from autogen_core import AgentId, AgentRuntime, TopicId, TypeSubscription
from message_envelope import MessageEnvelope

logger = logging.getLogger(__name__)

"""
A thin adapter that wraps/unwraps messages in `MessageEnvelope`
while delegating all real work to an inner `AgentRuntime`.

runtime = EnvelopedRuntime(inner=InProcessRuntime())
await runtime.start()
...
"""


class EnvelopedRuntime(AgentRuntime):
    """A potential drop-in replacement for any concrete AgentRuntime.

    It intercepts send/publish calls, wraps payloads inside `MessageEnvelope`,
    records telemetry, then forwards the raw payload on to the inner runtime
    so existing subscriptions continue to match on original message classes.
    """

    def __init__(self, *, inner: AgentRuntime) -> None:
        """Initialize the EnvelopedRuntime with an inner runtime."""
        self._inner: AgentRuntime = inner
        super().__init__()

    def __getattr__(self, item: str) -> Any:  # pragma: no cover
        """Delegate any missing methods to the inner runtime."""
        return getattr(self._inner, item)

    # region Public API - Send/Publish

    async def send_message(self, payload: Any, recipient: AgentId, **kwargs) -> None:
        """Wrap the payload in a MessageEnvelope and send it to the recipient."""
        env = self._make_envelope(payload, **kwargs)
        await self._log_envelope(env, recipient=recipient)
        # deliver original payload so existing handlers match their TypeSubscription
        await self._inner.send_message(env.payload, recipient)

    async def publish_message(self, payload: Any, topic: TopicId, **kwargs) -> None:
        """Wrap the payload in a MessageEnvelope and publish it to the topic."""
        env = self._make_envelope(payload, **kwargs)
        await self._log_envelope(env, topic=topic)
        await self._inner.publish_message(env.payload, topic)

    # endregion

    # region Public API - Subscriptions

    async def add_subscription(self, subscription: TypeSubscription) -> None:
        """Add a subscription to the inner runtime."""
        await self._inner.add_subscription(subscription)

    @staticmethod
    def _make_envelope(payload: Any, **kw) -> MessageEnvelope[Any]:
        return MessageEnvelope(
            payload=payload,
            ttl_ms=kw.get("ttl_ms"),
            headers=kw.get("headers"),
            correlation_id=kw.get("correlation_id"),
            causation_id=kw.get("causation_id"),
        )

    async def _log_envelope(self, env: MessageEnvelope[Any], **context) -> None:
        """Log the envelope and its context.

        Simple async hook you can extend to push OpenTelemetry spans,
        metrics, or write to a message some store.
        """
        logger.debug(f"Envelope {env} dispatched with context={context}")

    # region Proxy methods to the inner runtime.

    async def start(self) -> None:
        """Start the inner runtime.

        Base `AgentRuntime` type doesn't have this method right now but just an illustration.

        Could be a protocol on runtime implementations.
        """
        await self._inner.start()

    async def stop_when_idle(self) -> None:
        """Stop the inner runtime when idle.

        Base `AgentRuntime` type doesn't have this method right now but just an illustration.
        """
        await self._inner.stop_when_idle()

    # endregion
