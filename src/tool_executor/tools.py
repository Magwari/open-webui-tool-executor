"""
Tool Dispatcher.

Maps tool call names to their execution functions.
Each tool receives parsed arguments and returns a JSON string result.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .client import client as openwebui_client
from .executor import execute_code as execute_code_sandbox

log = logging.getLogger(__name__)


# ─── Tool Registry ───────────────────────────────


async def dispatch_tool(tool_name: str, arguments: dict) -> str:
    """
    Dispatch a tool call to the appropriate handler.

    Args:
        tool_name: Name of the tool to execute.
        arguments: Parsed JSON arguments from the LLM tool call.

    Returns:
        JSON string result from the tool execution.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return json.dumps({'error': f'Unknown tool: {tool_name}'})

    try:
        # Handler can be a coroutine function or a regular callable
        result = handler(**arguments)

        # If it's a coroutine, await it
        if hasattr(result, '__await__'):
            result = await result

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        return result

    except Exception as e:
        log.exception(f'Tool execution failed for {tool_name}: {e}')
        return json.dumps({'error': f'Tool {tool_name} failed: {str(e)}'})


# ─── Time Tools ────────────────────────────────

async def get_current_timestamp() -> str:
    now = datetime.now(timezone.utc)
    return json.dumps({
        'current_timestamp': int(now.timestamp()),
        'current_iso': now.isoformat(),
    })


async def calculate_timestamp(
    days_ago: int = 0, weeks_ago: int = 0,
    months_ago: int = 0, years_ago: int = 0
) -> str:
    from dateutil.relativedelta import relativedelta
    now = datetime.now(timezone.utc)
    delta = relativedelta(
        days=days_ago + weeks_ago * 7,
        months=months_ago,
        years=years_ago,
    )
    adjusted = now - delta
    return json.dumps({
        'current_timestamp': int(now.timestamp()),
        'current_iso': now.isoformat(),
        'calculated_timestamp': int(adjusted.timestamp()),
        'calculated_iso': adjusted.isoformat(),
    })


# ─── Async wrapper helper ──────────────────────

def _wrap_async(method: Callable) -> Callable:
    """Wrap an async method call into a coroutine-returning callable."""
    async def _call(**kwargs):
        return await method(**kwargs)
    _call.__name__ = method.__name__
    return _call


# ─── Tool Handler Mappings ─────────────────────
# Maps tool_name -> async callable
# Arguments are passed as keyword args from the LLM.

TOOL_HANDLERS: dict[str, Callable] = {
    # Time (local)
    'get_current_timestamp': get_current_timestamp,
    'calculate_timestamp': calculate_timestamp,

    # Code Execution (local thread pool)
    'execute_code': execute_code_sandbox,

    # Memory (via OpenWebUI API)
    'search_memories': _wrap_async(openwebui_client.search_memories),
    'add_memory': _wrap_async(openwebui_client.add_memory),
    'replace_memory_content': _wrap_async(openwebui_client.update_memory),
    'delete_memory': _wrap_async(openwebui_client.delete_memory),
    'list_memories': _wrap_async(openwebui_client.list_memories),

    # Knowledge Base (via OpenWebUI API)
    'list_knowledge_bases': _wrap_async(openwebui_client.list_knowledge_bases),
    'search_knowledge_bases': _wrap_async(openwebui_client.search_knowledge_bases),
    'search_knowledge_files': _wrap_async(openwebui_client.search_knowledge_files),
    'view_file': _wrap_async(openwebui_client.view_file),
    'view_knowledge_file': _wrap_async(openwebui_client.view_file),

    # Web Search (via OpenWebUI API)
    'search_web': _wrap_async(openwebui_client.search_web),
    'fetch_url': _wrap_async(openwebui_client.fetch_url),

    # Image Generation (via OpenWebUI API)
    'generate_image': _wrap_async(openwebui_client.generate_image),

    # Notes (via OpenWebUI API)
    'search_notes': _wrap_async(openwebui_client.search_notes),
    'view_note': _wrap_async(openwebui_client.view_note),
    'write_note': _wrap_async(openwebui_client.create_note),

    # Chat Search (via OpenWebUI API)
    'search_chats': _wrap_async(openwebui_client.search_chats),
    'view_chat': _wrap_async(openwebui_client.view_chat),

    # Skills (via OpenWebUI API)
    'view_skill': _wrap_async(openwebui_client.view_skill),

    # Channels (via OpenWebUI API)
    'search_channels': _wrap_async(openwebui_client.search_channels),

    # Calendar (via OpenWebUI API)
    'search_calendar_events': _wrap_async(openwebui_client.search_calendar_events),
    'create_calendar_event': _wrap_async(openwebui_client.create_calendar_event),
    'delete_calendar_event': _wrap_async(openwebui_client.delete_calendar_event),

    # Automation (via OpenWebUI API)
    'create_automation': _wrap_async(openwebui_client.create_automation),
    'list_automations': _wrap_async(openwebui_client.list_automations),
    'toggle_automation': _wrap_async(openwebui_client.toggle_automation),
    'delete_automation': _wrap_async(openwebui_client.delete_automation),
}


def get_available_tools() -> list[dict]:
    """Return list of registered tool names for debugging."""
    return [{'type': 'function', 'name': name} for name in TOOL_HANDLERS.keys()]