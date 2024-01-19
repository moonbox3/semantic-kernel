# Copyright (c) Microsoft. All rights reserved.

from typing import Any, ClassVar, Dict, List, Optional, Union
import re
from pydantic import Field, field_validator, root_validator
from semantic_kernel.sk_pydantic import SKBaseModel


class KernelJsonSchema(SKBaseModel):
    pass
