"""Tests for RAN environment."""
import pytest
from src.environment.ran_env import RANEnvironment

def test_environment_creation():
    env = RANEnvironment(n_cells=5)
    assert env.n_cells == 5

def test_reset():
    env = RANEnvironment()
    obs, info = env.reset()
    assert obs is not None
