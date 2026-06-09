import logging
import os

import pytest
import ray


@pytest.fixture(scope="session")
def ray_session():
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # to avoid a warning
    ray.init(num_cpus=2, log_to_driver=False, logging_level=logging.WARNING)
    yield
    ray.shutdown()
