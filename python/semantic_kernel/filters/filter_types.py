# Copyright (c) Microsoft. All rights reserved.

from enum import Enum


class FilterTypes(str, Enum):
    """Enum for the filter types."""

    AUTO_FUNCTION_INVOCATION = "auto_function_invocation"
    FUNCTION_INVOCATION = "function_invocation"
    PROMPT_RENDERING = "prompt_rendering"
