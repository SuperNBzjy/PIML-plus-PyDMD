"""Pytest fixtures used across the test-suite."""

from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture
def mocker():
    """Provide a lightweight substitute for the pytest-mock fixture."""

    yield mock
