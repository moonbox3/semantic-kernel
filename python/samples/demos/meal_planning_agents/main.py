# Copyright (c) Microsoft. All rights reserved.

import asyncio

from autogen_core import SingleThreadedAgentRuntime

from samples.demos.meal_planning_agents.agents.meal_planning_agent import MealPlanningAgent
from samples.demos.meal_planning_agents.custom_patterns.loop_with_user import LoopWithUserManager
from semantic_kernel.agents.orchestration.group_chat import GroupChatOrchestration


async def main():
    meal_planning_agent = MealPlanningAgent()

    user_loop_pattern = GroupChatOrchestration(
        manager=LoopWithUserManager(),
        agents=[meal_planning_agent],
    )

    await user_loop_pattern.start(
        task="Create a meal plan for next week for a family of two adults.",
        runtime=SingleThreadedAgentRuntime(),
    )


if __name__ == "__main__":
    asyncio.run(main())
