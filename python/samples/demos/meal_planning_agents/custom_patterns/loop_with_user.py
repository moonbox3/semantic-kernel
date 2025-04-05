# Copyright (c) Microsoft. All rights reserved.

import sys

from semantic_kernel.agents.orchestration.group_chat import BoolWithReason, ObjectWithReason, RoundRobinGroupChatManager
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover


class LoopWithUserManager(RoundRobinGroupChatManager):
    def __init__(self):
        super().__init__(user_input_func=self._user_input)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BoolWithReason:
        if len(chat_history) > 1:
            return BoolWithReason(
                value=True,
                reason="User input is required.",
            )

        return BoolWithReason(
            value=False,
            reason="User input is not required yet.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BoolWithReason:
        if (
            len(chat_history) > 0
            and chat_history.messages[-1].role == AuthorRole.USER
            and "approve" in chat_history.messages[-1].content.lower()
        ):
            return BoolWithReason(
                value=True,
                reason="User approved the plan.",
            )

        return BoolWithReason(
            value=False,
            reason="User approval is required.",
        )

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> ObjectWithReason:
        """Filter the results of the group chat."""
        return ObjectWithReason(
            value=chat_history.messages[-2],
            reason="The second to last message is the message from the assistant with the meal plan.",
        )

    async def _user_input(self, chat_history: ChatHistory) -> str:
        """Request user input."""
        print(f"Latest message: {chat_history.messages[-1].content}")
        return input("Your feedback (Approve if the plan is good) > ")
