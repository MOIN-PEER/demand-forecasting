import pytest

from demand_forecasting.features import build_features
from demand_forecasting.generate_data import generate


@pytest.fixture(scope="session")
def data():
    return generate(seed=7)


@pytest.fixture(scope="session")
def feat(data):
    return build_features(data)
