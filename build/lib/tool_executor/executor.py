"""
Code Executor - Thread pool based Python sandbox.

Executes Python code in a restricted environment using a thread pool,
preventing blocking of the async event loop.
"""

import asyncio
import builtins
import io
import logging
import sys
import textwrap
import concurrent.futures
from typing import Any, Optional

from .config import settings

log = logging.getLogger(__name__)

# Thread pool for code execution
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=settings.CODE_EXEC_MAX_WORKERS
)

# Blocked modules that cannot be imported during code execution
BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'socket', 'http', 'urllib',
    'requests', 'aiohttp', 'shutil', 'tempfile', 'pathlib',
}


def _restrict_imports(globals_dict: dict) -> None:
    """Block dangerous module imports."""
    _real_import = builtins.__import__

    def restricted_import(name, *args, **kwargs):
        top_level = name.split('.')[0]
        if top_level in BLOCKED_MODULES:
            raise ImportError(
                f"Import of '{name}' is blocked for security reasons."
            )
        return _real_import(name, *args, **kwargs)

    builtins.__import__ = restricted_import
    globals_dict['__builtins__'] = builtins


def execute_code_sync(code: str) -> dict[str, Any]:
    """
    Execute Python code synchronously in a sandboxed environment.

    Returns:
        dict with 'stdout', 'stderr', 'result' keys.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer

    local_ns = {}

    try:
        # Compile to catch syntax errors before execution
        compiled = compile(code, '<string>', 'exec')

        # Restrict imports for safety
        _restrict_imports(local_ns)

        exec(compiled, {'__builtins__': builtins}, local_ns)

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        # Try to get the last expression result
        result = local_ns.get('_result')
        if result is not None:
            result = str(result)
        else:
            result = None

        return {
            'status': 'success',
            'stdout': stdout,
            'stderr': stderr,
            'result': result,
        }

    except SyntaxError as e:
        return {
            'status': 'error',
            'stdout': stdout_buffer.getvalue(),
            'stderr': f'SyntaxError: {e.msg} (line {e.lineno})',
            'result': None,
        }
    except ImportError as e:
        return {
            'status': 'error',
            'stdout': stdout_buffer.getvalue(),
            'stderr': f'ImportError: {e}',
            'result': None,
        }
    except Exception as e:
        return {
            'status': 'error',
            'stdout': stdout_buffer.getvalue(),
            'stderr': f'{type(e).__name__}: {e}',
            'result': None,
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


async def execute_code(code: str, timeout: Optional[int] = None) -> str:
    """
    Execute Python code asynchronously using a thread pool.

    Args:
        code: Python code to execute.
        timeout: Execution timeout in seconds (default: from config).

    Returns:
        JSON string with execution results.
    """
    if timeout is None:
        timeout = settings.CODE_EXEC_TIMEOUT

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_thread_pool, execute_code_sync, code),
            timeout=timeout,
        )
        return _format_result(result)

    except asyncio.TimeoutError:
        return _format_result({
            'status': 'error',
            'stdout': '',
            'stderr': f'Execution timed out after {timeout} seconds.',
            'result': None,
        })
    except Exception as e:
        log.exception(f'Code execution failed: {e}')
        return _format_result({
            'status': 'error',
            'stdout': '',
            'stderr': f'Executor error: {e}',
            'result': None,
        })


def _format_result(result: dict) -> str:
    """Format execution result as JSON string."""
    import json
    return json.dumps(result, ensure_ascii=False)


def shutdown() -> None:
    """Shutdown the thread pool."""
    _thread_pool.shutdown(wait=False)