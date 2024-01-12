# Copyright (c) Microsoft. All rights reserved.

from semantic_kernel.sk_pydantic import SKBaseModel
from abc import ABC
import re

from pydantic import Field, field_validator, model_validator

class KernelPluginBase(SKBaseModel, ABC):

    name: str
    description: str

    @field_validator('name', mode="after")
    @classmethod
    def name_must_be_valid(cls, v: str) -> str:
        if ' ' not in v:
            raise ValueError('must contain a space')
        return v.title()