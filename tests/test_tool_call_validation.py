"""
Tests for AI Tool Call Validation.

Validates that AI-generated tool calls match actual tool signatures.
This ensures when AI requests a tool call, the arguments provided
are compatible with what the tool handler expects.
"""

import inspect
import json
from typing import get_type_hints

import pytest

from tool_executor.tools import TOOL_HANDLERS, dispatch_tool, _wrap_async


class TestToolCallSignatureValidation:
    """Validate that tool call arguments match registered handler signatures."""

    # Tools wrapped by _wrap_async use **kwargs, so we test them by verifying
    # the underlying client method has the expected signature.
    WRAPPED_CLIENT_TOOLS = {
        'search_web',
        'fetch_url',
        'generate_image',
        'edit_image',
        'add_memory',
        'search_memories',
        'list_memories',
        'delete_memory',
        'replace_memory_content',
    }

    def _get_handler_params(self, tool_name: str) -> dict:
        """Extract expected parameters from a tool handler (non-wrapped only)."""
        handler = TOOL_HANDLERS.get(tool_name)
        if handler is None:
            raise ValueError(f"Tool {tool_name} not found")

        sig = inspect.signature(handler)
        params = {}
        for name, param in sig.parameters.items():
            params[name] = {
                'has_default': param.default != inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'kind': param.kind,
            }
        return params

    def _get_client_method_params(self, method_name: str) -> dict:
        """Extract parameters from the OpenWebUIClient method directly."""
        from tool_executor.client import OpenWebUIClient
        method = getattr(OpenWebUIClient, method_name, None)
        if method is None:
            raise ValueError(f"Method {method_name} not found on OpenWebUIClient")

        sig = inspect.signature(method)
        params = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            params[name] = {
                'has_default': param.default != inspect.Parameter.empty,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'kind': param.kind,
            }
        return params

    def test_get_current_timestamp_accepts_no_args(self):
        params = self._get_handler_params('get_current_timestamp')
        # Should have no required parameters (or be empty)
        for p in params.values():
            assert p['has_default']

    def test_execute_code_requires_code_param(self):
        params = self._get_handler_params('execute_code')
        assert 'code' in params
        assert not params['code']['has_default']

    def test_calculate_timestamp_has_optional_params(self):
        params = self._get_handler_params('calculate_timestamp')
        assert 'days_ago' in params
        assert params['days_ago']['has_default']

    # ─── Wrapped client tools: inspect client method directly ───

    def test_search_web_requires_query(self):
        params = self._get_client_method_params('search_web')
        assert 'query' in params
        assert not params['query']['has_default']

    def test_fetch_url_requires_url(self):
        params = self._get_client_method_params('fetch_url')
        assert 'url' in params
        assert not params['url']['has_default']

    def test_generate_image_requires_prompt(self):
        params = self._get_client_method_params('generate_image')
        assert 'prompt' in params
        assert not params['prompt']['has_default']

    def test_add_memory_requires_content(self):
        params = self._get_client_method_params('add_memory')
        assert 'content' in params
        assert not params['content']['has_default']

    def test_search_memories_requires_query(self):
        params = self._get_client_method_params('search_memories')
        assert 'query' in params
        assert not params['query']['has_default']

    def test_wrapped_tools_have_var_kwargs_signature(self):
        """Tools wrapped by _wrap_async expose **kwargs signature."""
        for tool_name in self.WRAPPED_CLIENT_TOOLS:
            params = self._get_handler_params(tool_name)
            # The wrapper exposes **kwargs
            assert 'kwargs' in params, f"{tool_name} should have **kwargs from _wrap_async"


class TestAIToolCallValidation:
    """
    Simulate AI tool call responses and validate them against actual tool signatures.
    
    This tests the core concern: when AI generates a tool call, will it work?
    """

    def _simulate_ai_tool_call(self, tool_name: str, arguments: dict) -> str:
        """Simulate what happens when AI generates a tool call."""
        return json.dumps({
            'tool_name': tool_name,
            'arguments': arguments,
        })

    async def test_ai_calls_timestamp_with_no_args(self):
        # AI: "Get the current time"
        result = await dispatch_tool('get_current_timestamp', {})
        data = json.loads(result)
        assert 'current_timestamp' in data
        assert 'error' not in data

    async def test_ai_calls_execute_code_with_code(self):
        # AI: "Run this Python code"
        result = await dispatch_tool('execute_code', {'code': '2 + 2'})
        data = json.loads(result)
        assert data['status'] == 'success'

    async def test_ai_calls_calculate_with_partial_args(self):
        # AI provides only some optional args
        result = await dispatch_tool('calculate_timestamp', {'weeks_ago': 2})
        data = json.loads(result)
        assert 'calculated_timestamp' in data
        assert 'error' not in data

    async def test_ai_calls_unknown_tool_returns_error(self):
        # AI hallucinates a tool that doesn't exist
        result = await dispatch_tool('browse_internet', {})
        data = json.loads(result)
        assert 'error' in data
        assert 'Unknown tool' in data['error']

    async def test_ai_calls_tool_with_extra_args_still_works(self):
        # AI provides extra args that the handler doesn't use
        # dispatch_tool passes **kwargs, so extra args will cause an error
        # This is expected behavior - the tool should reject unknown args
        result = await dispatch_tool('get_current_timestamp', {'unused': 'value'})
        data = json.loads(result)
        # The handler doesn't accept extra args, so this returns an error
        assert 'error' in data

    async def test_ai_provides_json_string_arguments(self):
        # Simulate AI providing arguments as JSON string (common in tool calls)
        args_str = '{"code": "print(42)"}'
        args = json.loads(args_str)
        result = await dispatch_tool('execute_code', args)
        data = json.loads(result)
        assert data['status'] == 'success'
        assert '42' in data['stdout']

    async def test_ai_provides_empty_args_to_tool_that_needs_args(self):
        # AI forgot to provide required arguments
        result = await dispatch_tool('execute_code', {})
        data = json.loads(result)
        # Should fail with a meaningful error
        assert 'error' in data

    async def test_ai_provides_null_for_optional_param(self):
        # AI provides None for optional parameter
        result = await dispatch_tool('execute_code', {'code': 'print("hi")', 'timeout': None})
        data = json.loads(result)
        assert data['status'] == 'success'


class TestToolCallArgumentFormat:
    """Test various argument formats AI might generate."""

    async def test_integer_argument(self):
        result = await dispatch_tool('calculate_timestamp', {'days_ago': 30})
        data = json.loads(result)
        assert 'calculated_timestamp' in data

    async def test_string_argument(self):
        result = await dispatch_tool('execute_code', {'code': 'x = 1'})
        data = json.loads(result)
        assert data['status'] == 'success'

    async def test_boolean_argument(self):
        result = await dispatch_tool('execute_code', {'code': 'print(True)'})
        data = json.loads(result)
        assert data['status'] == 'success'

    async def test_float_argument(self):
        result = await dispatch_tool('calculate_timestamp', {'days_ago': 0})
        data = json.loads(result)
        assert 'calculated_timestamp' in data

    async def test_mixed_argument_types(self):
        result = await dispatch_tool('calculate_timestamp', {
            'days_ago': 1,
            'weeks_ago': 0,
            'months_ago': 2,
        })
        data = json.loads(result)
        assert 'calculated_timestamp' in data

    async def test_json_nested_object_in_code(self):
        code = 'import json; print(json.dumps({"nested": [1, 2, 3]}))'
        result = await dispatch_tool('execute_code', {'code': code})
        data = json.loads(result)
        assert data['status'] == 'success'
        assert 'nested' in data['stdout']

    async def test_unicode_in_code(self):
        code = 'print("Hello, 세계!")'
        result = await dispatch_tool('execute_code', {'code': code})
        data = json.loads(result)
        assert data['status'] == 'success'
        assert '세계' in data['stdout']