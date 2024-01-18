# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Union

from pydantic import Field

from semantic_kernel.orchestration.sk_function import SKFunction
from semantic_kernel.plugin_definition import constants
from semantic_kernel.plugin_definition.kernel_plugin import KernelPlugin

from semantic_kernel.sk_pydantic import SKBaseModel

if TYPE_CHECKING:
    from semantic_kernel.orchestration.sk_function_base import SKFunctionBase

logger: logging.Logger = logging.getLogger(__name__)


class KernelPluginCollection(SKBaseModel):
    plugins: Dict[str, KernelPlugin] = Field(default_factory=dict)

    def __init__(self, plugins=None):
        self.plugins = {}
        if plugins is not None:
            if isinstance(plugins, KernelPluginCollection):
                self.plugins = {plugin.name: plugin for plugin in plugins.plugins.values()}
            else:
                self.add_range(plugins)
        super().__init__(plugins=self.plugins)

    def add(self, plugin):
        if plugin is None or plugin.name is None:
            raise ValueError("Plugin and plugin.name must not be None")
        self.plugins[plugin.name] = plugin

    def add_range(self, plugins):
        if plugins is None:
            raise ValueError("Plugins must not be None")
        for plugin in plugins:
            self.add(plugin)

    def remove(self, plugin):
        if plugin is None or plugin.name is None:
            return False
        return self.plugins.pop(plugin.name, None) is not None
    
    def get_plugin(self, name):
        return self.plugins.get(name, None)

    def clear(self):
        self.plugins.clear()

    def __iter__(self):
        return iter(self.plugins.values())

    def __len__(self):
        return len(self.plugins)

    def contains(self, plugin):
        if plugin is None or plugin.name is None:
            return False
        return self.plugins.get(plugin.name) == plugin

    def __getitem__(self, name):
        if name not in self.plugins:
            raise KeyError(f"Plugin {name} not found.")
        return self.plugins[name]
