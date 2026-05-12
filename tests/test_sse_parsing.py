"""
Tests for SSE parsing and accumulation logic.

Covers:
- _parse_sse_data
- _accumulate_streaming_response
- Tool call merging across chunks
"""

import json

from tool_executor.main import _parse_sse_data, _accumulate_streaming_response


class TestParseSseData:
    def test_single_event(self):
        sse = 'data: {"key": "value"}\n\n'
        result = _parse_sse_data(sse)

        assert len(result) == 1
        assert result[0]['key'] == 'value'

    def test_multiple_events(self):
        sse = '''data: {"a": 1}\n\ndata: {"b": 2}\n\n'''
        result = _parse_sse_data(sse)

        assert len(result) == 2
        assert result[0]['a'] == 1
        assert result[1]['b'] == 2

    def test_done_event_ignored(self):
        sse = 'data: [DONE]\n\n'
        result = _parse_sse_data(sse)

        assert len(result) == 0

    def test_empty_data_ignored(self):
        sse = 'data: \n\n'
        result = _parse_sse_data(sse)

        assert len(result) == 0

    def test_invalid_json_ignored(self):
        sse = 'data: not json\n\n'
        result = _parse_sse_data(sse)

        assert len(result) == 0

    def test_mixed_valid_invalid(self):
        sse = '''data: {"ok": true}\n\ndata: invalid\n\ndata: {"ok": false}\n\n'''
        result = _parse_sse_data(sse)

        assert len(result) == 2
        assert result[0]['ok'] is True
        assert result[1]['ok'] is False

    def test_data_with_leading_spaces(self):
        sse = 'data:   {"spaced": true}\n\n'
        result = _parse_sse_data(sse)

        assert len(result) == 1
        assert result[0]['spaced'] is True


class TestAccumulateStreamingResponse:
    def test_empty_chunks(self):
        message, finish = _accumulate_streaming_response([])
        assert message['role'] == 'assistant'
        assert message['content'] == ''
        assert finish is None

    def test_content_accumulation(self):
        chunks = [
            {'choices': [{'delta': {'role': 'assistant', 'content': 'Hello '}}]},
            {'choices': [{'delta': {'content': 'World'}}]},
        ]
        message, _ = _accumulate_streaming_response(chunks)

        assert message['content'] == 'Hello World'
        assert message['role'] == 'assistant'

    def test_role_extraction(self):
        chunks = [
            {'choices': [{'delta': {'role': 'assistant'}}]},
            {'choices': [{'delta': {'content': 'test'}}]},
        ]
        message, _ = _accumulate_streaming_response(chunks)

        assert message['role'] == 'assistant'

    def test_finish_reason_extraction(self):
        chunks = [
            {'choices': [{'delta': {'content': 'done'}, 'finish_reason': 'stop'}]},
        ]
        message, finish = _accumulate_streaming_response(chunks)

        assert finish == 'stop'

    def test_tool_call_merging(self):
        # Simulate streaming tool call split across chunks
        chunks = [
            {
                'choices': [{
                    'delta': {
                        'role': 'assistant',
                        'tool_calls': [{
                            'index': 0,
                            'id': 'call_abc',
                            'type': 'function',
                            'function': {'name': 'get_current_timestamp', 'arguments': ''},
                        }],
                    },
                }],
            },
            {
                'choices': [{
                    'delta': {
                        'tool_calls': [{
                            'index': 0,
                            'function': {'arguments': '{}'},
                        }],
                    },
                }],
            },
        ]
        message, _ = _accumulate_streaming_response(chunks)

        assert 'tool_calls' in message
        assert len(message['tool_calls']) == 1
        assert message['tool_calls'][0]['function']['name'] == 'get_current_timestamp'
        assert message['tool_calls'][0]['function']['arguments'] == '{}'

    def test_multiple_tool_calls(self):
        chunks = [
            {
                'choices': [{
                    'delta': {
                        'tool_calls': [
                            {'index': 0, 'id': 'call_1', 'type': 'function', 'function': {'name': 'tool_a', 'arguments': '{}'}},
                            {'index': 1, 'id': 'call_2', 'type': 'function', 'function': {'name': 'tool_b', 'arguments': '{}'}},
                        ],
                    },
                }],
            },
        ]
        message, _ = _accumulate_streaming_response(chunks)

        assert len(message['tool_calls']) == 2

    def test_reasoning_accumulation(self):
        chunks = [
            {'choices': [{'delta': {'reasoning': 'Thinking '}}]},
            {'choices': [{'delta': {'reasoning': '...'}}]},
            {'choices': [{'delta': {'content': 'Answer'}}]},
        ]
        message, _ = _accumulate_streaming_response(chunks)

        assert message.get('reasoning') == 'Thinking ...'
        assert message['content'] == 'Answer'

    def test_tool_calls_finish_reason(self):
        chunks = [
            {
                'choices': [{
                    'delta': {
                        'tool_calls': [{
                            'index': 0,
                            'id': 'call_x',
                            'type': 'function',
                            'function': {'name': 'test', 'arguments': '{}'},
                        }],
                    },
                    'finish_reason': 'tool_calls',
                }],
            },
        ]
        message, finish = _accumulate_streaming_response(chunks)

        assert finish == 'tool_calls'
        assert message['tool_calls'][0]['function']['name'] == 'test'