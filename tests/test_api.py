"""
Tests for FastAPI Endpoints.

Uses TestClient for simple endpoints and httpx_mock for chat completions.
"""

import json

import pytest
from fastapi.testclient import TestClient

from tool_executor.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get('/health')
        assert response.status_code == 200

        data = response.json()
        assert data['status'] == 'ok'
        assert data['version'] == '1.0.0'


class TestListToolsEndpoint:
    def test_list_tools_returns_list(self):
        response = client.get('/tools')
        assert response.status_code == 200

        data = response.json()
        assert 'tools' in data
        assert isinstance(data['tools'], list)
        assert len(data['tools']) > 0

    def test_tool_format(self):
        response = client.get('/tools')
        tools = response.json()['tools']

        for tool in tools:
            assert tool['type'] == 'function'
            assert 'name' in tool
            assert isinstance(tool['name'], str)

    def test_expected_tools_present(self):
        response = client.get('/tools')
        tool_names = [t['name'] for t in response.json()['tools']]

        assert 'get_current_timestamp' in tool_names
        assert 'execute_code' in tool_names
        assert 'search_web' in tool_names