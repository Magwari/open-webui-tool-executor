"""
Tests for Tool Dispatcher.

Covers:
- dispatch_tool with valid/invalid tool names
- Argument parsing (JSON string vs dict)
- Error handling in handlers
"""

import json

import pytest

from tool_executor.tools import dispatch_tool, TOOL_HANDLERS, get_available_tools


class TestDispatchTool:
    async def test_dispatch_known_tool(self):
        result = await dispatch_tool('get_current_timestamp', {})
        data = json.loads(result)

        assert 'current_timestamp' in data
        assert 'error' not in data

    async def test_dispatch_unknown_tool(self):
        result = await dispatch_tool('nonexistent_tool', {})
        data = json.loads(result)

        assert 'error' in data
        assert 'Unknown tool' in data['error']

    async def test_dispatch_execute_code(self):
        result = await dispatch_tool('execute_code', {'code': 'print(42)'})
        data = json.loads(result)

        assert data['status'] == 'success'
        assert '42' in data['stdout']

    async def test_dispatch_calculate_timestamp(self):
        result = await dispatch_tool('calculate_timestamp', {'days_ago': 3})
        data = json.loads(result)

        assert 'calculated_timestamp' in data
        assert 'error' not in data

    async def test_dispatch_with_empty_args(self):
        # get_current_timestamp doesn't require args
        result = await dispatch_tool('get_current_timestamp', {})
        data = json.loads(result)

        assert 'current_timestamp' in data

    async def test_dispatch_result_is_json_string(self):
        result = await dispatch_tool('get_current_timestamp', {})
        # Result should be a valid JSON string
        data = json.loads(result)
        assert isinstance(data, dict)


class TestToolRegistry:
    def test_registry_has_tools(self):
        assert len(TOOL_HANDLERS) > 0

    def test_get_available_tools_format(self):
        tools = get_available_tools()
        assert isinstance(tools, list)

        for tool in tools:
            assert 'type' in tool
            assert 'name' in tool
            assert tool['type'] == 'function'

    def test_expected_tools_registered(self):
        expected = [
            'get_current_timestamp',
            'calculate_timestamp',
            'execute_code',
            'search_memories',
            'search_web',
            'generate_image',
        ]

        for name in expected:
            assert name in TOOL_HANDLERS, f"{name} should be registered"

    def test_all_tools_have_callable_handlers(self):
        for name, handler in TOOL_HANDLERS.items():
            assert callable(handler), f"{name} handler must be callable"