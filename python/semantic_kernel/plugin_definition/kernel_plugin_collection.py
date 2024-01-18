# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

from pydantic import Field, root_validator

from semantic_kernel.plugin_definition import constants
from semantic_kernel.plugin_definition.kernel_plugin import KernelPlugin
from semantic_kernel.plugin_definition.default_kernel_plugin import DefaultKernelPlugin

from semantic_kernel.sk_pydantic import SKBaseModel

#if TYPE_CHECKING:
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase

logger: logging.Logger = logging.getLogger(__name__)


class KernelPluginCollection(SKBaseModel):
    GLOBAL_PLUGIN: ClassVar[str] = constants.GLOBAL_PLUGIN
    plugins: Optional[Dict[str, KernelPlugin]] = Field(default_factory=dict)

    @root_validator(pre=True)
    def process_plugins(cls, values):
        plugins_input = values.get("plugins")
        if isinstance(plugins_input, KernelPluginCollection):
            # Extract plugins from another KernelPluginCollection instance
            values["plugins"] = {plugin.name: plugin for plugin in plugins_input.plugins.values()}
        elif isinstance(plugins_input, (list, set, tuple)):
            # Process an iterable of plugins
            plugins_dict = {}
            for plugin in plugins_input:
                if plugin is None or plugin.name is None:
                    raise ValueError("Plugin and plugin.name must not be None")
                if plugin.name in plugins_dict:
                    raise ValueError(f"Duplicate plugin name detected: {plugin.name}")
                plugins_dict[plugin.name] = plugin
            values["plugins"] = plugins_dict
        return values

    def add(self, plugin: Union[KernelPlugin, SKFunctionBase]) -> None:
        """
        Add a single plugin to the collection

        Args:
            plugin (KernelPlugin): The plugin to add to the collection.

        Raises:
            ValueError: If the plugin or plugin.name is None.
        """
        if plugin is None or plugin.name is None:
            raise ValueError("Plugin and plugin.name must not be None")
        if isinstance(plugin, SKFunctionBase):
            plugin = DefaultKernelPlugin.from_function(plugin)
        if plugin.name in self.plugins:
            raise ValueError(f"Plugin with name {plugin.name} already exists")
        self.plugins[plugin.name] = plugin

    def add_range(self, plugins: List[KernelPlugin]) -> None:
        """
        Add a list of plugins to the collection

        Args:
            plugins (List[KernelPlugin]): The plugins to add to the collection.

        Raises:
            ValueError: If the plugins list is None.
        """

        if plugins is None:
            raise ValueError("Plugins must not be None")
        for plugin in plugins:
            self.add(plugin)

    def remove(self, plugin: KernelPlugin) -> bool:
        """
        Remove a plugin from the collection

        Args:
            plugin (KernelPlugin): The plugin to remove from the collection.

        Returns:
            True if the plugin was removed, False otherwise.
        """
        if plugin is None or plugin.name is None:
            return False
        return self.plugins.pop(plugin.name, None) is not None

    def get_plugin(self, name: str) -> Optional[KernelPlugin]:
        """
        Get a plugin from the collection

        Args:
            name (str): The name of the plugin to retrieve.

        Returns:
            The plugin if it exists, None otherwise.
        """
        return self.plugins.get(name, None)

    def clear(self):
        """Clear the collection of all plugins"""
        self.plugins.clear()

    def __iter__(self) -> Any:
        """Define an iterator for the collection"""
        return iter(self.plugins.values())

    def __len__(self) -> int:
        """Define the length of the collection"""
        return len(self.plugins)

    def contains(self, plugin) -> bool:
        """Check if the collection contains a plugin"""
        if plugin is None or plugin.name is None:
            return False
        return self.plugins.get(plugin.name) == plugin

    def __getitem__(self, name):
        """Define the [] operator for the collection

        Args:
            name (str): The name of the plugin to retrieve.

        Returns:
            The plugin if it exists, None otherwise.

        Raises:
            KeyError: If the plugin does not exist.
        """
        if name not in self.plugins:
            raise KeyError(f"Plugin {name} not found.")
        return self.plugins[name]
