"""Shared fixtures for the engine test suite.

Loads the real model once per test session (it's the whole point — these tests
exist to catch regressions in the actual hook engine, not a mock of it). Uses
whatever `MODEL_NAME` is configured in `app/model.py`; expects the weights to
already be present in the local HuggingFace cache (see README "Run it locally").
"""

import pytest

from app.model import XRayModel
from app.xray_engine import XRayHookEngine


@pytest.fixture(scope="session")
def xray_model() -> XRayModel:
    model = XRayModel()
    model.load()
    return model


@pytest.fixture(scope="session")
def engine(xray_model: XRayModel) -> XRayHookEngine:
    return XRayHookEngine(xray_model)
