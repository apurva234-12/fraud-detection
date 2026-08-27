import pytest
from core.config import Config


@pytest.fixture
def config():
    return Config()
