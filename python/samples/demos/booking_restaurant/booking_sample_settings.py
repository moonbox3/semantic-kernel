# Copyright (c) Microsoft. All rights reserved.

from typing import ClassVar

from pydantic import SecretStr

from semantic_kernel.kernel_pydantic import KernelBaseSettings


class BookingSampleSettings(KernelBaseSettings):
    """Restaurant Booking Sample settings

    The settings are first loaded from environment variables with the prefix 'BOOKING_'. If the
    environment variables are not found, the settings can be loaded from a .env file with the
    encoding 'utf-8'. If the settings are not found in the .env file, the settings are ignored;
    however, validation will fail alerting that the settings are missing.

    Required settings for prefix 'BOOKING_' are:
    - client_id = The App Registration Client ID (Env var BOOKING_CLIENT_ID)
    - tenant_id = The App Registration Tenant ID (Env var BOOKING_TENANT_ID)
    - client_secret = The App Registration Client Secret (Env var BOOKING_CLIENT_SECRET)
    - business_id = The sample booking service ID (Env var BOOKING_BUSINESS_ID)
    - service_id = The sample booking service ID (Env var BOOKING_SERVICE_ID)

    For more information on these required settings, please see the sample's README.md file.
    """

    env_prefix: ClassVar[str] = "BOOKING_SAMPLE_"

    client_id: str
    tenant_id: str
    client_secret: SecretStr
    business_id: str
    service_id: str
