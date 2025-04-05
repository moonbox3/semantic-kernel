# Copyright (c) Microsoft. All rights reserved.

from enum import Enum

from semantic_kernel.kernel_pydantic import KernelBaseModel


class MealType(Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class Meal(KernelBaseModel):
    meal_type: MealType
    recipe_name: str


class DailyMealPlan(KernelBaseModel):
    meals: list[Meal]


class WeeklyMealPlan(KernelBaseModel):
    daily_meal_plans: list[DailyMealPlan]
