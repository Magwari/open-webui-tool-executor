"""
Tests for local tool handlers (no external API dependency).

Covers:
- get_current_timestamp
- calculate_timestamp
- execute_code (sandbox)
"""

import json
from datetime import datetime, timezone

import pytest

from tool_executor.tools import get_current_timestamp, calculate_timestamp
from tool_executor.executor import execute_code


# ─── get_current_timestamp ───

class TestGetCurrentTimestamp:
    async def test_returns_valid_timestamp(self):
        result = await get_current_timestamp()
        data = json.loads(result)

        assert 'current_timestamp' in data
        assert 'current_iso' in data
        assert isinstance(data['current_timestamp'], int)
        assert isinstance(data['current_iso'], str)

    async def test_timestamp_is_recent(self):
        result = await get_current_timestamp()
        data = json.loads(result)

        now = int(datetime.now(timezone.utc).timestamp())
        # Timestamp should be within 60 seconds of now
        assert abs(data['current_timestamp'] - now) < 60

    async def test_iso_format_is_valid(self):
        result = await get_current_timestamp()
        data = json.loads(result)

        # Should parse as valid ISO format
        parsed = datetime.fromisoformat(data['current_iso'])
        assert parsed.tzinfo is not None


# ─── calculate_timestamp ───

class TestCalculateTimestamp:
    async def test_no_offset_returns_current(self):
        result = await calculate_timestamp()
        data = json.loads(result)

        assert 'calculated_timestamp' in data
        assert 'calculated_iso' in data
        assert data['current_timestamp'] == data['calculated_timestamp']

    async def test_days_ago(self):
        result = await calculate_timestamp(days_ago=7)
        data = json.loads(result)

        now = int(datetime.now(timezone.utc).timestamp())
        diff = now - data['calculated_timestamp']
        # 7 days = 604800 seconds, allow ±1 day tolerance
        assert 6 * 86400 <= diff <= 8 * 86400

    async def test_months_ago(self):
        result = await calculate_timestamp(months_ago=1)
        data = json.loads(result)

        assert data['current_timestamp'] != data['calculated_timestamp']
        assert data['calculated_timestamp'] < data['current_timestamp']

    async def test_years_ago(self):
        result = await calculate_timestamp(years_ago=1)
        data = json.loads(result)

        now = int(datetime.now(timezone.utc).timestamp())
        diff = now - data['calculated_timestamp']
        # 1 year ≈ 365 days, allow ±30 days tolerance
        assert 335 * 86400 <= diff <= 395 * 86400

    async def test_combined_offset(self):
        result = await calculate_timestamp(days_ago=1, weeks_ago=1, months_ago=1)
        data = json.loads(result)

        assert data['calculated_timestamp'] < data['current_timestamp']

    async def test_response_contains_all_fields(self):
        result = await calculate_timestamp(days_ago=5)
        data = json.loads(result)

        assert 'current_timestamp' in data
        assert 'current_iso' in data
        assert 'calculated_timestamp' in data
        assert 'calculated_iso' in data


# ─── execute_code ───

class TestExecuteCode:
    async def test_simple_print(self):
        result = await execute_code('print("hello")')
        data = json.loads(result)

        assert data['status'] == 'success'
        assert 'hello' in data['stdout']

    async def test_expression_result(self):
        code = 'x = 1 + 2\nprint(x)'
        result = await execute_code(code)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert '3' in data['stdout']

    async def test_syntax_error(self):
        result = await execute_code('def incomplete')
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'SyntaxError' in data['stderr']

    async def test_runtime_error(self):
        result = await execute_code('1 / 0')
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'ZeroDivisionError' in data['stderr']

    async def test_blocked_module_os(self):
        result = await execute_code('import os')
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'ImportError' in data['stderr']

    async def test_blocked_module_subprocess(self):
        result = await execute_code('import subprocess')
        data = json.loads(result)

        assert data['status'] == 'error'

    async def test_blocked_module_socket(self):
        result = await execute_code('import socket')
        data = json.loads(result)

        assert data['status'] == 'error'

    async def test_allowed_module_math(self):
        code = 'import math\nprint(math.sqrt(16))'
        result = await execute_code(code)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert '4.0' in data['stdout']

    async def test_allowed_module_json(self):
        code = 'import json\nprint(json.dumps({"a": 1}))'
        result = await execute_code(code)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert '"a"' in data['stdout']

    async def test_empty_code(self):
        result = await execute_code('')
        data = json.loads(result)

        assert data['status'] == 'success'

    async def test_multiline_code(self):
        code = '''
for i in range(3):
    print(i)
'''
        result = await execute_code(code)
        data = json.loads(result)

        assert data['status'] == 'success'
        assert '0' in data['stdout']
        assert '1' in data['stdout']
        assert '2' in data['stdout']

    async def test_stderr_capture(self):
        import warnings
        code = '''
import warnings
warnings.warn("test warning")
print("done")
'''
        result = await execute_code(code)
        data = json.loads(result)

        assert data['status'] == 'success'

    async def test_timeout(self):
        # Infinite loop should timeout with a short timeout
        result = await execute_code('while True: pass', timeout=2)
        data = json.loads(result)

        assert data['status'] == 'error'
        assert 'timed out' in data['stderr'].lower()