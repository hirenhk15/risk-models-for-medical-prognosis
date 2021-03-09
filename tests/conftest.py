# This file will initialize Flask app

import os
import sys
sys.path.append(os.path.join('../', os.getcwd()))

import pytest
import tempfile
from app import app as flask_app

@pytest.fixture
def app():
    """Returns flask app"""
    yield flask_app

@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()