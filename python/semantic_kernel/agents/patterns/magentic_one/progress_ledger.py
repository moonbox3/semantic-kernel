# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.kernel_pydantic import KernelBaseModel


class ProgressLedgerItem(KernelBaseModel):
    """A progress ledger item."""

    reason: str
    answer: str | bool


class ProgressLedger(KernelBaseModel):
    """A progress ledger."""

    is_request_satisfied: ProgressLedgerItem
    is_in_loop: ProgressLedgerItem
    is_progress_being_made: ProgressLedgerItem
    next_speaker: ProgressLedgerItem
    instruction_or_question: ProgressLedgerItem
