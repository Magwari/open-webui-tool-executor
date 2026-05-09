"""
Tool Executor Server - Main Application.

FastAPI server that provides OpenAI-compatible /chat/completions endpoint
with automatic builtin tool execution via SSE streaming.

Flow:
  1. Receive OpenAI-format chat completion request
  2. Forward to OpenWebUI with streaming
  3. Forward LLM tokens to client via SSE in real-time
  4. On tool_calls: execute tools, append results to messages, retry
  5. On stop: send final SSE [DONE] event
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .config import settings
from .client import OpenWebUIClient
from .models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    Delta,
    Message,
    ToolCall,
    ToolCallFunction,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from .tools import dispatch_tool
from .executor import shutdown as executor_shutdown

log = logging.getLogger(__name__)

# ─── App Setup ──────────────────────────

app = FastAPI(
    title='OpenWebUI Tool Executor',
    description='Proxy server that executes builtin tools for OpenWebUI LLM responses.',
    version='1.0.0',
)

# Global client instance
ow_client = OpenWebUIClient()


@app.on_event('shutdown')
async def shutdown_event():
    await ow_client.close()
    executor_shutdown()


# ─── SSE Helpers ────────────────────────

def _sse_line(data: str) -> str:
    """Format data as an SSE line."""
    return f'data: {data}\n\n'


def _make_chunk(
    model: str,
    content: Optional[str] = None,
    role: Optional[str] = None,
    tool_calls: Optional[list[ToolCall]] = None,
    finish_reason: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> str:
    """Create an OpenAI-compatible SSE chunk."""
    chunk_id = chat_id or f'chatcmpl-{int(time.time())}'
    delta = Delta(
        role=role,
        content=content,
        tool_calls=tool_calls,
    )
    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[Choice(delta=delta, finish_reason=finish_reason)],
    )
    return _sse_line(chunk.model_dump_json())


def _make_tool_start_event(tool_id: str, tool_name: str, arguments: str) -> str:
    event = ToolCallStartEvent(
        tool_id=tool_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    return _sse_line(event.model_dump_json())


def _make_tool_result_event(tool_id: str, tool_name: str, result: str, error: Optional[str] = None) -> str:
    event = ToolCallResultEvent(
        tool_id=tool_id,
        tool_name=tool_name,
        result=result,
        error=error,
    )
    return _sse_line(event.model_dump_json())


# ─── SSE Stream Generator ─────────────────


async def chat_stream(
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """
    Main SSE generator implementing the function calling loop.

    For each iteration:
      1. Send chat completion request to OpenWebUI (streaming)
      2. Forward LLM tokens to client in real-time
      3. If tool_calls returned, execute them and append results
      4. Repeat until finish_reason is 'stop' or max iterations reached
    """
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    model_id = request.model
    chat_id = request.chat_id or f'local-{int(time.time())}'

    iteration = 0
    max_iterations = settings.MAX_TOOL_ITERATIONS

    while iteration < max_iterations:
        iteration += 1
        log.info(f'Iteration {iteration}/{max_iterations}, messages count: {len(messages)}')

        # Send request to OpenWebUI with streaming
        response = await ow_client.chat_completions(
            model=model_id,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            stop=request.stop,
            chat_id=chat_id,
            session_id=request.session_id,
        )

        # Parse the non-streaming response
        if not response or 'choices' not in response:
            yield _sse_line(json.dumps({'error': 'Invalid response from OpenWebUI'}))
            return

        choice = response['choices'][0] if response['choices'] else None
        if not choice:
            yield _sse_line(json.dumps({'error': 'No choices in response'}))
            return

        message = choice.get('message', {})
        finish_reason = choice.get('finish_reason', '')

        # Send role delta first if needed
        if message.get('role') and not messages[-1].get('role') == 'assistant':
            yield _make_chunk(model=model_id, role=message.get('role'), chat_id=chat_id)

        # Send content
        content = message.get('content', '')
        if content:
            # Split content into chunks for realistic streaming feel
            for chunk in _split_content(content):
                yield _make_chunk(model=model_id, content=chunk, chat_id=chat_id)

        # Check for tool calls
        tool_calls = message.get('tool_calls', [])

        if tool_calls and (finish_reason == 'tool_calls' or tool_calls):
            # Send tool call chunks
            for tc in tool_calls:
                tc_func = tc.get('function', {})
                tool_call_obj = ToolCall(
                    id=tc.get('id', ''),
                    function=ToolCallFunction(
                        name=tc_func.get('name', ''),
                        arguments=tc_func.get('arguments', '{}'),
                    ),
                )
                yield _make_chunk(
                    model=model_id,
                    tool_calls=[tool_call_obj],
                    chat_id=chat_id,
                )

            # Execute each tool call and collect results
            tool_results = []
            for tc in tool_calls:
                tc_func = tc.get('function', {})
                tool_name = tc_func.get('name', '')
                tool_id = tc.get('id', '')
                arguments_json = tc_func.get('arguments', '{}')

                # Send tool start event
                yield _make_tool_start_event(tool_id, tool_name, arguments_json)
                log.info(f'Executing tool: {tool_name} (id: {tool_id})')

                # Parse arguments
                try:
                    if isinstance(arguments_json, str):
                        args = json.loads(arguments_json)
                    else:
                        args = arguments_json
                except json.JSONDecodeError:
                    args = {}

                # Execute tool
                try:
                    result = await dispatch_tool(tool_name, args)
                    error = None
                except Exception as e:
                    result = ''
                    error = str(e)
                    log.exception(f'Tool {tool_name} execution failed: {e}')

                # Send tool result event
                yield _make_tool_result_event(tool_id, tool_name, result, error)

                # Store result for later
                tool_results.append({
                    'tool_call_id': tool_id,
                    'tool_name': tool_name,
                    'content': result,
                })

            # Append assistant message with tool calls (once)
            messages.append({
                'role': 'assistant',
                'content': content or None,
                'tool_calls': tool_calls,
            })

            # Append tool result messages
            for tr in tool_results:
                messages.append({
                    'role': 'tool',
                    'tool_call_id': tr['tool_call_id'],
                    'content': tr['content'],
                    'name': tr['tool_name'],
                })

            # Continue loop for next iteration
            continue

        # No more tool calls - final response
        yield _make_chunk(model=model_id, finish_reason=finish_reason or 'stop', chat_id=chat_id)
        yield _sse_line('[DONE]')
        return

    # Max iterations reached
    log.warning(f'Max tool iterations ({max_iterations}) reached.')
    yield _make_chunk(model=model_id, finish_reason='stop', chat_id=chat_id)
    yield _sse_line('[DONE]')


def _split_content(content: str, chunk_size: int = 20) -> list[str]:
    """Split content into smaller chunks for streaming effect."""
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunks.append(content[i:i + chunk_size])
    return chunks if chunks else ['']


# ─── Endpoints ────────────────────────

@app.post('/chat/completions')
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with builtin tool execution.

    Accepts standard OpenAI format requests and automatically executes
    tool calls returned by the LLM, forwarding all events via SSE.
    """
    if not request.stream:
        # Non-streaming: execute and return final result
        messages = [m.model_dump(exclude_none=True) for m in request.messages]
        chat_id = request.chat_id or f'local-{int(time.time())}'

        iteration = 0
        max_iterations = settings.MAX_TOOL_ITERATIONS

        while iteration < max_iterations:
            iteration += 1
            response = await ow_client.chat_completions(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                chat_id=chat_id,
                session_id=request.session_id,
            )

            choice = response.get('choices', [{}])[0]
            message = choice.get('message', {})
            finish_reason = choice.get('finish_reason', '')
            tool_calls = message.get('tool_calls', [])

            if tool_calls and finish_reason == 'tool_calls':
                messages.append(message)
                for tc in tool_calls:
                    tc_func = tc.get('function', {})
                    tool_name = tc_func.get('name', '')
                    tool_id = tc.get('id', '')
                    arguments_json = tc_func.get('arguments', '{}')

                    try:
                        if isinstance(arguments_json, str):
                            args = json.loads(arguments_json)
                        else:
                            args = arguments_json
                        result = await dispatch_tool(tool_name, args)
                    except Exception as e:
                        result = f'Error: {e}'

                    messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_id,
                        'content': result,
                        'name': tool_name,
                    })
            else:
                return response

        return response

    # Streaming response
    return StreamingResponse(
        chat_stream(request),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        },
    )


@app.get('/health')
async def health():
    return {'status': 'ok', 'version': '1.0.0'}


@app.get('/tools')
async def list_tools():
    from .tools import get_available_tools
    return {'tools': get_available_tools()}


# ─── Run ────────

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        'main:app',
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
    )