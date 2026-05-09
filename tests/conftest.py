import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def test_csv():
    return os.path.join(os.path.dirname(__file__), "data", "test.csv")



