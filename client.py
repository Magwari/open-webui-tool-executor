"""
OpenWebUI HTTP Client.

Handles all HTTP communication with the OpenWebUI backend:
- Streaming chat completions (SSE)
- Tool execution via REST API endpoints
- Authentication via API Key
"""

import json
import logging
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

import aiohttp
from aiohttp import ClientTimeout

from .config import settings

log = logging.getLogger(__name__)


class OpenWebUIClient:
    """Async HTTP client for OpenWebUI API."""

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None and self._owns_session:
            timeout = ClientTimeout(total=settings.OPENWEBUI_TIMEOUT)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @property
    def headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {settings.OPENWEBUI_API_KEY}',
            'Content-Type': 'application/json',
        }

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and self._owns_session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ─── Chat Completions (Streaming) ─────────────────

    async def chat_completions_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Any] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[Any] = None,
        chat_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream chat completions from OpenWebUI via SSE.

        Yields raw SSE lines (bytes) from the response stream.
        """
        payload = {
            'model': model,
            'messages': messages,
            'stream': True,
        }

        # Optional parameters
        if temperature is not None:
            payload['temperature'] = temperature
        if max_tokens is not None:
            payload['max_tokens'] = max_tokens
        if top_p is not None:
            payload['top_p'] = top_p
        if stop is not None:
            payload['stop'] = stop
        if tools is not None:
            payload['tools'] = tools
        if tool_choice is not None:
            payload['tool_choice'] = tool_choice
        if chat_id:
            payload['chat_id'] = chat_id
        if session_id:
            payload['session_id'] = session_id

        url = f'{settings.OPENWEBUI_BASE_URL}/api/chat/completions'

        try:
            async with self.session.post(
                url,
                json=payload,
                headers=self.headers,
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    log.error(f'OpenWebUI chat error {response.status}: {error_body}')
                    raise Exception(f'OpenWebUI error: {response.status} - {error_body}')

                # Stream SSE chunks
                async for line in response.content:
                    if line:
                        yield line

        except aiohttp.ClientError as e:
            log.error(f'Failed to connect to OpenWebUI: {e}')
            raise

    # ─── Chat Completions (Non-streaming) ─────────────

    async def chat_completions(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Get non-streaming chat completion from OpenWebUI."""
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            **kwargs,
        }

        url = f'{settings.OPENWEBUI_BASE_URL}/api/chat/completions'

        async with self.session.post(
            url,
            json=payload,
            headers=self.headers,
        ) as response:
            if response.status != 200:
                error_body = await response.text()
                raise Exception(f'OpenWebUI error: {response.status} - {error_body}')
            return await response.json()

    # ─── Tool Execution APIs ──────────────────────────

    async def _get(self, path: str, params: Optional[dict] = None) -> Any:
        url = f'{settings.OPENWEBUI_BASE_URL}{path}'
        async with self.session.get(url, params=params, headers=self.headers) as r:
            if r.status >= 400:
                text = await r.text()
                raise Exception(f'GET {path} failed ({r.status}): {text}')
            return await r.json()

    async def _post(self, path: str, json_data: Optional[dict] = None) -> Any:
        url = f'{settings.OPENWEBUI_BASE_URL}{path}'
        async with self.session.post(
            url, json=json_data or {}, headers=self.headers
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise Exception(f'POST {path} failed ({r.status}): {text}')
            data = await r.read()
            if data:
                return json.loads(data)
            return None

    async def _put(self, path: str, json_data: Optional[dict] = None) -> Any:
        url = f'{settings.OPENWEBUI_BASE_URL}{path}'
        async with self.session.put(
            url, json=json_data or {}, headers=self.headers
        ) as r:
            if r.status >= 400:
                text = await r.text()
                raise Exception(f'PUT {path} failed ({r.status}): {text}')
            data = await r.read()
            if data:
                return json.loads(data)
            return None

    async def _delete(self, path: str) -> Any:
        url = f'{settings.OPENWEBUI_BASE_URL}{path}'
        async with self.session.delete(url, headers=self.headers) as r:
            if r.status >= 400:
                text = await r.text()
                raise Exception(f'DELETE {path} failed ({r.status}): {text}')
            data = await r.read()
            if data:
                return json.loads(data)
            return True

    # ─── Memory Tools ────────────────────────

    async def search_memories(self, content: str, k: int = 5) -> str:
        result = await self._post('/api/v1/memories/query', {'content': content, 'k': k})
        return self._format_memory_result(result)

    async def add_memory(self, content: str) -> str:
        result = await self._post('/api/v1/memories/add', {'content': content})
        return json.dumps({'status': 'success', 'id': result.get('id') if result else None})

    async def update_memory(self, memory_id: str, content: str) -> str:
        result = await self._post(f'/api/v1/memories/{memory_id}/update', {'content': content})
        return json.dumps({'status': 'success', 'id': result.get('id') if result else None})

    async def delete_memory(self, memory_id: str) -> str:
        result = await self._delete(f'/api/v1/memories/{memory_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})

    async def list_memories(self) -> str:
        result = await self._get('/api/v1/memories')
        if isinstance(result, list):
            memories = []
            for m in result:
                memories.append({
                    'id': m.get('id'),
                    'content': m.get('content'),
                    'created_at': m.get('created_at'),
                })
            return json.dumps(memories)
        return json.dumps([])

    @staticmethod
    def _format_memory_result(result: Any) -> str:
        if not result:
            return json.dumps([])
        if hasattr(result, 'documents'):
            memories = []
            docs = result.get('documents', [[]])[0] if result.get('documents') else []
            metas = result.get('metadatas', [[]])[0] if result.get('metadatas') else []
            ids = result.get('ids', [[]])[0] if result.get('ids') else []
            for idx, doc in enumerate(docs):
                memories.append({
                    'id': ids[idx] if idx < len(ids) else None,
                    'content': doc,
                    'created_at': metas[idx].get('created_at', 'Unknown') if idx < len(metas) else 'Unknown',
                })
            return json.dumps(memories)
        return json.dumps(result)

    # ─── Knowledge Base Tools ────────────────────

    async def list_knowledge_bases(self, page: int = 1) -> str:
        result = await self._get(f'/api/v1/knowledge?page={page}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {
                'id': kb.get('id'),
                'name': kb.get('name'),
                'description': kb.get('description', ''),
                'file_count': len(kb.get('files', [])) if kb.get('files') else 0,
            }
            for kb in items
        ])

    async def search_knowledge_bases(self, query: str, page: int = 1) -> str:
        result = await self._get(f'/api/v1/knowledge/search?query={query}&page={page}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {
                'id': kb.get('id'),
                'name': kb.get('name'),
                'description': kb.get('description', ''),
                'file_count': len(kb.get('files', [])) if kb.get('files') else 0,
            }
            for kb in items
        ])

    async def search_knowledge_files(self, query: str, page: int = 1) -> str:
        result = await self._get(f'/api/v1/knowledge/search/files?query={query}&page={page}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {'id': f.get('id'), 'filename': f.get('filename')}
            for f in items
        ])

    async def view_file(self, file_id: str) -> str:
        result = await self._get(f'/api/v1/files/{file_id}')
        if result:
            return json.dumps({
                'id': result.get('id'),
                'filename': result.get('filename'),
                'content': result.get('data', {}).get('content', '') if result.get('data') else '',
            })
        return json.dumps({'error': 'File not found'})

    # ─── Web Search Tools ──────────────────────

    async def search_web(self, query: str, max_results: Optional[int] = None) -> str:
        result = await self._post('/api/v1/retrieval/web/search', {
            'query': query,
            'max_results': max_results or 5,
        })
        if result:
            return json.dumps(result if isinstance(result, list) else [result])
        return json.dumps([])

    async def fetch_url(self, url: str) -> str:
        result = await self._post('/api/v1/retrieval/web/loader', {'url': url})
        if result:
            content = result.get('content', '') if isinstance(result, dict) else str(result)
            return content
        return ''

    # ─── Image Generation Tools ───────────────

    async def generate_image(self, prompt: str, size: str = '512x512', steps: int = 50) -> str:
        result = await self._post('/api/v1/images/generations', {
            'prompt': prompt,
            'size': size,
            'steps': steps,
        })
        if result:
            return json.dumps({'status': 'success', 'images': result if isinstance(result, list) else [result]})
        return json.dumps({'error': 'Failed to generate image'})

    # ─── Notes Tools ───────────────────────────

    async def search_notes(self, query: str, limit: int = 5) -> str:
        result = await self._get(f'/api/v1/notes/search?query={query}&limit={limit}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {'id': n.get('id'), 'title': n.get('title'), 'snippet': n.get('data', {}).get('content', '')[:200] if n.get('data') else ''}
            for n in items
        ])

    async def view_note(self, note_id: str) -> str:
        result = await self._get(f'/api/v1/notes/{note_id}')
        if result:
            return json.dumps({
                'id': result.get('id'),
                'title': result.get('title'),
                'content': result.get('data', {}).get('content', '') if result.get('data') else '',
            })
        return json.dumps({'error': 'Note not found'})

    async def create_note(self, title: str, content: str) -> str:
        result = await self._post('/api/v1/notes', {
            'title': title,
            'data': {'content': content},
        })
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id')})
        return json.dumps({'error': 'Failed to create note'})

    # ─── Chat Search Tools ────────────────────

    async def search_chats(self, query: str, limit: int = 5) -> str:
        result = await self._get(f'/api/v1/chats/search?query={query}&limit={limit}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {'id': c.get('id'), 'title': c.get('title'), 'updated_at': c.get('updated_at')}
            for c in items
        ])

    async def view_chat(self, chat_id: str) -> str:
        result = await self._get(f'/api/v1/chats/{chat_id}')
        if result:
            history = result.get('chat', {}).get('history', {}) if result.get('chat') else {}
            messages = history.get('messages', {}) if history else {}
            return json.dumps({
                'id': result.get('id'),
                'title': result.get('title'),
                'message_count': len(messages),
            })
        return json.dumps({'error': 'Chat not found'})

    # ─── Skills Tools ──────────────────────

    async def view_skill(self, skill_id: str) -> str:
        result = await self._get(f'/api/v1/skills/{skill_id}')
        if result:
            return json.dumps({
                'name': result.get('name'),
                'content': result.get('content', result.get('data', '')),
            })
        return json.dumps({'error': f'Skill {skill_id} not found'})

    # ─── Channel Tools ──────────────────────

    async def search_channels(self, query: str, limit: int = 5) -> str:
        result = await self._get(f'/api/v1/channels/search?query={query}&limit={limit}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {'id': c.get('id'), 'name': c.get('name'), 'description': c.get('description', '')}
            for c in items
        ])

    # ─── Calendar Tools ────────────────────

    async def search_calendar_events(
        self, query: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, limit: int = 10
    ) -> str:
        params = {}
        if query:
            params['query'] = query
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        params['limit'] = limit
        result = await self._get('/api/v1/calendars/events', params)
        items = result.get('items', result.get('events', [])) if result else []
        return json.dumps(items[:limit])

    async def create_calendar_event(
        self, title: str, start: str, end: Optional[str] = None, description: Optional[str] = None
    ) -> str:
        data = {'title': title, 'start': start}
        if end:
            data['end'] = end
        if description:
            data['description'] = description
        result = await self._post('/api/v1/calendars/events', data)
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'title': title})
        return json.dumps({'error': 'Failed to create event'})

    async def delete_calendar_event(self, event_id: str) -> str:
        result = await self._delete(f'/api/v1/calendars/events/{event_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})

    # ─── Automation Tools ────────────────

    async def create_automation(self, name: str, prompt: str, rrule: str) -> str:
        result = await self._post('/api/v1/automations', {
            'name': name,
            'data': {'prompt': prompt, 'rrule': rrule},
        })
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'name': name})
        return json.dumps({'error': 'Failed to create automation'})

    async def list_automations(self, limit: int = 10) -> str:
        result = await self._get(f'/api/v1/automations?limit={limit}')
        items = result.get('items', []) if result else []
        return json.dumps([
            {'id': a.get('id'), 'name': a.get('name'), 'is_active': a.get('is_active', True)}
            for a in items
        ])

    async def toggle_automation(self, automation_id: str) -> str:
        result = await self._post(f'/api/v1/automations/{automation_id}/toggle')
        return json.dumps({'status': 'success', 'id': automation_id})

    async def delete_automation(self, automation_id: str) -> str:
        result = await self._delete(f'/api/v1/automations/{automation_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})


# Singleton client instance
client = OpenWebUIClient()