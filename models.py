"""
Pydantic models for OpenAI-compatible Chat Completion API.

Defines request/response structures for the /chat/completions endpoint,
including tool calling support and SSE streaming format.
"""

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ─── Request Models ────────────────────────────────────────────────


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    id: str
    type: Literal['function'] = 'function'
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal['system', 'user', 'assistant', 'tool']
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = True
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stop: Optional[Any] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    # OpenWebUI-specific fields
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    parent_id: Optional[str] = None


# ─── Response Models (OpenAI-compatible chunks) ────────────────────


class Delta(BaseModel):
    role: Optional[Literal['assistant', 'tool']] = None
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    index: int = 0
    delta: Delta
    finish_reason: Optional[Literal['stop', 'tool_calls', 'length']] = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f'chatcmpl-{uuid.uuid4().hex[:8]}')
    object: Literal['chat.completion.chunk'] = 'chat.completion.chunk'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ''
    choices: list[Choice]


# ─── SSE Event Models (custom tool events) ─────────────────────────


class ToolCallStartEvent(BaseModel):
    type: Literal['tool_call_start'] = 'tool_call_start'
    tool_id: str
    tool_name: str
    arguments: str


class ToolCallResultEvent(BaseModel):
    type: Literal['tool_call_result'] = 'tool_call_result'
    tool_id: str
    tool_name: str
    result: str
    error: Optional[str] = None


# ─── Internal Models ──────────────────────────────────────────────


class ToolExecResult(BaseModel):
    """Result from executing a single tool call."""
    tool_id: str
    tool_name: str
    result: str
    error: Optional[str] = None


class OpenWebUIResponse(BaseModel):
    """Parsed response from OpenWebUI chat/completions."""
    id: Optional[str] = None
    model: Optional[str] = None
    choices: Optional[list[dict]] = None
    # For non-streaming responses
    message: Optional[Message] = None
    # For streaming: accumulated state
    accumulated_content: str = ''
    accumulated_tool_calls: list[ToolCall] = []
    finish_reason: Optional[str] = None