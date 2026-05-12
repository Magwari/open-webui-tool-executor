"""
Tool Executor Server - Main Application.

FastAPI server that provides OpenAI-compatible /chat/completions endpoint
with automatic builtin tool execution via SSE streaming.

Flow:
  1. Receive OpenAI-format chat completion request
  2. Forward streaming request to OpenWebUI
  3. Forward SSE events to client in real-time (passthrough)
  4. Accumulate delta chunks to detect tool_calls, reasoning, content
  5. On tool_calls: execute tools, append results to messages, retry
  6. On stop: send final SSE [DONE] event
"""

import asyncio
import json
import logging
import re
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

# ─── App Setup ───

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


# ─── SSE Helpers ───

def _sse_line(data: str) -> str:
    """Format data as an SSE line."""
    return f'data: {data}\n\n'


def _make_chunk(
    model: str,
    content: Optional[str] = None,
    reasoning: Optional[str] = None,
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
        reasoning=reasoning,
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


# ─── SSE Parser Helpers ───

_SSE_DATA_RE = re.compile(r'^data:\s*(.+)$', re.MULTILINE)


def _parse_sse_data(sse_text: str) -> list[dict]:
    """
    Parse SSE text blob into list of parsed JSON chunks.
    Handles multiple SSE events in one chunk.
    """
    results = []
    for match in _SSE_DATA_RE.finditer(sse_text):
        data = match.group(1).strip()
        if not data or data == '[DONE]':
            continue
        try:
            parsed = json.loads(data)
            results.append(parsed)
        except json.JSONDecodeError:
            continue
    return results


def _accumulate_streaming_response(chunks: list[dict]) -> tuple:
    """
    Accumulate OpenAI streaming chunks into a complete message structure.
    
    Extracts delta fields: role, content, reasoning, tool_calls, finish_reason.
    
    Returns:
        (message_dict, finish_reason)
    """
    full_content = []
    full_reasoning = []
    all_tool_calls = []
    last_role = None
    finish_reason = None

    for chunk in chunks:
        choices = chunk.get('choices')
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get('delta', {})
        fr = choice.get('finish_reason')
        if fr:
            finish_reason = fr

        if delta.get('role'):
            last_role = delta['role']

        # Accumulate content
        if delta.get('content') is not None:
            full_content.append(delta['content'])

        # Accumulate reasoning
        if delta.get('reasoning') is not None:
            full_reasoning.append(delta['reasoning'])

        # Accumulate tool calls
        if delta.get('tool_calls'):
            for tc in delta['tool_calls']:
                all_tool_calls.append(tc)

    # Merge tool calls by index (streaming splits them across chunks)
    merged_tool_calls = {}
    for tc in all_tool_calls:
        idx = tc.get('index', 0)
        if idx not in merged_tool_calls:
            merged_tool_calls[idx] = {
                'id': '',
                'type': 'function',
                'function': {'name': '', 'arguments': ''},
            }
        if tc.get('id'):
            merged_tool_calls[idx]['id'] = tc['id']
        if tc.get('function'):
            if tc['function'].get('name'):
                merged_tool_calls[idx]['function']['name'] = tc['function']['name']
            if tc['function'].get('arguments') is not None:
                merged_tool_calls[idx]['function']['arguments'] += tc['function']['arguments']

    message: dict = {
        'role': last_role or 'assistant',
        'content': ''.join(full_content),
    }

    # Include reasoning if present
    if full_reasoning:
        message['reasoning'] = ''.join(full_reasoning)

    if merged_tool_calls:
        message['tool_calls'] = list(merged_tool_calls.values())

    return message, finish_reason


# ─── SSE Stream Generator ───

async def chat_stream(
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """
    Main SSE generator implementing the function calling loop.

    For each iteration:
      1. Send streaming chat completion request to OpenWebUI
      2. Forward SSE events to client in real-time (passthrough)
      3. Accumulate response to detect tool_calls / reasoning
      4. If tool_calls returned, execute them and append results
      5. Repeat until finish_reason is 'stop' or max iterations reached
    """
    messages = [m.model_dump(exclude_none=True) for m in request.messages]
    model_id = request.model
    chat_id = request.chat_id or f'local-{int(time.time())}'

    iteration = 0
    max_iterations = settings.MAX_TOOL_ITERATIONS

    while iteration < max_iterations:
        iteration += 1
        log.info(f'Iteration {iteration}/{max_iterations}, messages count: {len(messages)}')

        # Accumulate chunks for tool_call detection
        accumulated_chunks: list[dict] = []

        # Stream SSE events from OpenWebUI and forward in real-time
        try:
            async for sse_text in ow_client.chat_completions_stream(
                model=model_id,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stop=request.stop,
                chat_id=chat_id,
                session_id=request.session_id,
            ):
                # Forward SSE data to client in real-time (passthrough)
                yield sse_text

                # Parse and accumulate chunks for tool_call detection
                chunks = _parse_sse_data(sse_text)
                accumulated_chunks.extend(chunks)
        except Exception as e:
            log.exception(f'Streaming error: {e}')
            yield _sse_line(json.dumps({'error': str(e)}))
            return

        # Accumulate response from chunks
        message, finish_reason = _accumulate_streaming_response(accumulated_chunks)

        if not message:
            yield _sse_line(json.dumps({'error': 'No message from OpenWebUI'}))
            return

        tool_calls = message.get('tool_calls', [])

        if tool_calls and finish_reason == 'tool_calls':
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
                    # Inject model_id for list_knowledge tool
                    if tool_name == 'list_knowledge':
                        args.setdefault('model_id', model_id)

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

            # Append assistant message with tool calls
            assistant_msg = {
                'role': 'assistant',
                'content': message.get('content') or None,
                'tool_calls': tool_calls,
            }
            # Include reasoning if present
            if message.get('reasoning'):
                assistant_msg['reasoning'] = message['reasoning']
            messages.append(assistant_msg)

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

        # No more tool calls - send final chunk
        yield _make_chunk(model=model_id, finish_reason=finish_reason or 'stop', chat_id=chat_id)
        yield _sse_line('[DONE]')
        return

    # Max iterations reached
    log.warning(f'Max tool iterations ({max_iterations}) reached.')
    yield _make_chunk(model=model_id, finish_reason='stop', chat_id=chat_id)
    yield _sse_line('[DONE]')


# ─── Endpoints ───

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

                        # Inject model_id for list_knowledge tool
                        if tool_name == 'list_knowledge':
                            args.setdefault('model_id', request.model)

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

    # Streaming response - SSE passthrough
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


# ─── Run ───

if __name__ == '__main__':
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        'tool_executor.main:app',
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=True,
    )