# Copyright (c) Microsoft. All rights reserved.


from samples.demos.meal_planning_agents.models.meal_plan import WeeklyMealPlan
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments


class MealPlanningAgent(ChatCompletionAgent):
    def __init__(self):
        super().__init__(
            name="meal_planning_agent",
            description="An agent that creates meal plans based on user preferences.",
            instructions=(
                "You are a meal planning assistant. You plan meals based on user preferences and dietary restrictions."
            ),
            service=OpenAIChatCompletion(),
            arguments=KernelArguments(
                settings=OpenAIChatPromptExecutionSettings(
                    response_format=WeeklyMealPlan,
                ),
            ),
        )
