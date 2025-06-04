# Copyright (c) Microsoft. All rights reserved.

import functools
import inspect
from typing import Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel, create_model

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ReturnT = TypeVar("ReturnT")


class FunctionTool(Generic[ArgsT, ReturnT]):
    """A tool that wraps a function, allowing it to be used as a callable tool in agents."""

    def __init__(
        self,
        func: Callable[..., Awaitable[ReturnT]],
        name: str,
        description: str,
        input_model: type[ArgsT],
    ):
        """Initialize the FunctionTool with a function, name, description, and input model."""
        self._func = func
        self.name = name
        self.description = description
        self.input_model = input_model

    def model_json_schema(self):
        """Return the JSON schema for the input model."""
        return self.input_model.model_json_schema()

    async def __call__(self, *args, **kwargs):
        """Invoke the function with the provided arguments."""
        return await self._func(*args, **kwargs)

    async def run(self, args: BaseModel):
        """Run the function with the provided arguments as a Pydantic model."""
        if inspect.iscoroutinefunction(self._func):
            return await self._func(**args.model_dump())
        # TODO(evmattso): handle synchronous functions
        return self._func(**args.model_dump())


def function_tool(func=None, *, name=None, description=None):
    """Decorator: Wrap a function as a FunctionTool and return the callable tool object."""

    def wrapper(f):
        tool_name = name or f.__name__
        tool_desc = description or (f.__doc__ or "")
        sig = inspect.signature(f)
        fields = {}
        for pname, param in sig.parameters.items():
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else str
            default = param.default if param.default is not inspect.Parameter.empty else ...
            fields[pname] = (ann, default)
        # Dynamically create a Pydantic model that validates this tool's input arguments
        InputModel = create_model(f"{tool_name}_input", **fields)

        # Return a callable FunctionTool object
        class CallableTool(FunctionTool):
            async def __call__(self, *args, **kwargs):
                return await f(*args, **kwargs)

        return functools.update_wrapper(
            CallableTool(f, tool_name, tool_desc, InputModel),
            f,
        )

    return wrapper(func) if func else wrapper
