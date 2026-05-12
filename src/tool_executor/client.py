"""
OpenWebUI HTTP Client.

Handles all HTTP communication with the OpenWebUI backend.
Function signatures match builtin.py tool parameters so that LLM-generated
tool_calls dispatch correctly via **kwargs.
"""

import json
import logging
from typing import Any, Optional

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

    # ─── Chat Completions (Non-streaming) ────

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

    # ─── Chat Completions (Streaming) ────

    async def chat_completions_stream(
        self,
        model: str,
        messages: list[dict],
        **kwargs,
    ):
        """
        Stream chat completion from OpenWebUI as SSE events.
        Yields raw SSE text lines from the response.
        """
        import aiohttp

        payload = {
            'model': model,
            'messages': messages,
            'stream': True,
            **kwargs,
        }

        url = f'{settings.OPENWEBUI_BASE_URL}/api/chat/completions'

        async with self.session.post(
            url,
            json=payload,
            headers={
                'Authorization': f'Bearer {settings.OPENWEBUI_API_KEY}',
            },
        ) as response:
            if response.status != 200:
                error_body = await response.text()
                raise Exception(f'OpenWebUI error: {response.status} - {error_body}')

            # Stream SSE events line by line
            async for line, _ in response.content.iter_chunks():
                text = line.decode('utf-8')
                yield text

    # ─── HTTP Helpers ────

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

    # ═══════════════════════════════════════════
    # Memory Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_memories(self, query: str, count: int = 5) -> str:
        """Search memories. builtin.py: search_memories(query, count=5)"""
        result = await self._post('/api/v1/memories/query', {
            'content': query,
            'k': count,
        })
        return self._format_memory_result(result)

    async def add_memory(self, content: str) -> str:
        """Add a memory. builtin.py: add_memory(content)"""
        result = await self._post('/api/v1/memories/add', {'content': content})
        return json.dumps({'status': 'success', 'id': result.get('id') if result else None})

    async def replace_memory_content(self, memory_id: str, content: str) -> str:
        """Update memory content. builtin.py: replace_memory_content(memory_id, content)"""
        result = await self._post(f'/api/v1/memories/{memory_id}/update', {'content': content})
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'content': result.get('content', '')})
        return json.dumps({'error': 'Failed to update memory'})

    async def delete_memory(self, memory_id: str) -> str:
        """Delete a memory. builtin.py: delete_memory(memory_id)"""
        result = await self._delete(f'/api/v1/memories/{memory_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})

    async def list_memories(self) -> str:
        """List all memories. builtin.py: list_memories()"""
        result = await self._get('/api/v1/memories')
        if isinstance(result, list):
            memories = []
            for m in result:
                memories.append({
                    'id': m.get('id'),
                    'content': m.get('content'),
                    'created_at': m.get('created_at'),
                    'updated_at': m.get('updated_at'),
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

    # ═══════════════════════════════════════════
    # Knowledge Base Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def list_knowledge_bases(self, count: int = 10, skip: int = 0) -> str:
        """List KBs. builtin.py: list_knowledge_bases(count=10, skip=0)"""
        result = await self._get(f'/api/v1/knowledge/search', params={
            'query': '',
            'skip': skip,
            'limit': count,
        })
        items = result.get('items', []) if result else []
        return json.dumps([
            {
                'id': kb.get('id'),
                'name': kb.get('name'),
                'description': kb.get('description', ''),
                'file_count': len(kb.get('files', [])) if kb.get('files') else 0,
                'updated_at': kb.get('updated_at'),
            }
            for kb in items
        ])

    async def search_knowledge_bases(self, query: str, count: int = 5, skip: int = 0) -> str:
        """Search KBs. builtin.py: search_knowledge_bases(query, count=5, skip=0)"""
        result = await self._get(f'/api/v1/knowledge/search', params={
            'query': query,
            'skip': skip,
            'limit': count,
        })
        items = result.get('items', []) if result else []
        return json.dumps([
            {
                'id': kb.get('id'),
                'name': kb.get('name'),
                'description': kb.get('description', ''),
                'file_count': len(kb.get('files', [])) if kb.get('files') else 0,
                'updated_at': kb.get('updated_at'),
            }
            for kb in items
        ])

    async def search_knowledge_files(self, query: str, knowledge_id: Optional[str] = None, count: int = 5, skip: int = 0) -> str:
        """Search KB files. builtin.py: search_knowledge_files(query, knowledge_id, count=5, skip=0)"""
        params: dict[str, Any] = {
            'query': query,
            'skip': skip,
            'limit': count,
        }
        if knowledge_id:
            params['knowledge_id'] = knowledge_id

        result = await self._get('/api/v1/knowledge/search/files', params=params)
        items = result.get('items', []) if result else []
        return json.dumps([
            {
                'id': f.get('id'),
                'filename': f.get('filename'),
                'knowledge_id': f.get('knowledge_id'),
                'knowledge_name': f.get('knowledge_name'),
                'updated_at': f.get('updated_at'),
            }
            for f in items
        ])

    async def view_file(self, file_id: str, offset: int = 0, max_chars: int = 10000) -> str:
        """View file content. builtin.py: view_file(file_id, offset=0, max_chars=10000)"""
        result = await self._get(f'/api/v1/files/{file_id}', params={
            'offset': offset,
            'max_chars': max_chars,
        })
        if result:
            return json.dumps({
                'id': result.get('id'),
                'filename': result.get('filename'),
                'content': result.get('content', ''),
                'truncated': result.get('truncated'),
                'total_chars': result.get('total_chars'),
                'returned_chars': result.get('returned_chars'),
                'next_offset': result.get('next_offset'),
            })
        return json.dumps({'error': 'File not found'})

    async def view_knowledge_file(self, file_id: str, offset: int = 0, max_chars: int = 10000) -> str:
        """View knowledge file content. builtin.py: view_knowledge_file(file_id, offset=0, max_chars=10000)"""
        # Same endpoint as view_file for now
        return await self.view_file(file_id, offset=offset, max_chars=max_chars)

    # ─── query_knowledge_files (vector search) ───
    # This requires embedding which cannot be done via simple REST API.
    # Return a placeholder error directing users to use the UI.

    async def query_knowledge_files(self, query: str, knowledge_ids: Optional[list] = None, count: int = 5) -> str:
        """
        Semantic search across knowledge files.
        NOTE: This requires the embedding function which is only available
        server-side in OpenWebUI. Returning an informative error.
        """
        return json.dumps({
            'error': 'query_knowledge_files requires server-side embedding and is not available through the proxy API. '
                     'Please use search_knowledge_files for filename-based search instead.'
        })

    # ─── query_knowledge_bases (semantic KB discovery) ───

    async def query_knowledge_bases(self, query: str, count: int = 5) -> str:
        """
        Semantic search for knowledge bases by meaning.
        NOTE: Requires server-side embedding. Returning an informative error.
        """
        return json.dumps({
            'error': 'query_knowledge_bases requires server-side embedding and is not available through the proxy API. '
                     'Please use search_knowledge_bases for text-based search instead.'
        })

    # ─── list_knowledge (attached model knowledge) ───

    async def list_knowledge(self, model_id: str = '') -> str:
        """
        List knowledge bases, files, and notes attached to the given model.

        Fetches model meta via /api/v1/models/model?id={model_id}, extracts
        knowledge base references, then resolves each KB's details and files.
        Also fetches user notes via /api/v1/notes.
        """
        result = {
            'knowledge_bases': [],
            'files': [],
            'notes': [],
        }

        # 1. Fetch model meta to get knowledge base references
        try:
            model_data = await self._get('/api/v1/models/model', params={'id': model_id})
        except Exception as e:
            log.warning(f'list_knowledge: failed to fetch model {model_id}: {e}')
            return json.dumps(result)

        if not model_data:
            return json.dumps(result)

        # 2. Extract knowledge base IDs from model meta
        meta = model_data.get('meta') or {}
        knowledge_refs = meta.get('knowledge') or []

        # 3. Resolve each knowledge base
        fetched_kb_ids = set()
        for kb_ref in knowledge_refs:
            kb_id = None
            if isinstance(kb_ref, dict):
                kb_id = kb_ref.get('id')
            elif isinstance(kb_ref, str):
                kb_id = kb_ref

            if not kb_id or kb_id in fetched_kb_ids:
                continue
            fetched_kb_ids.add(kb_id)

            try:
                kb_data = await self._get(f'/api/v1/knowledge/{kb_id}')
                if not kb_data:
                    continue

                result['knowledge_bases'].append({
                    'id': kb_data.get('id'),
                    'name': kb_data.get('name'),
                    'description': kb_data.get('description', ''),
                })

                # Extract files from knowledge base
                kb_files = kb_data.get('files') or []
                for f in kb_files:
                    result['files'].append({
                        'id': f.get('id'),
                        'filename': f.get('filename'),
                        'knowledge_id': kb_id,
                    })
            except Exception as e:
                log.warning(f'list_knowledge: failed to fetch knowledge {kb_id}: {e}')
                continue

        # 4. Fetch user notes
        try:
            notes_data = await self._get('/api/v1/notes')
            if isinstance(notes_data, list):
                for n in notes_data:
                    result['notes'].append({
                        'id': n.get('id'),
                        'title': n.get('title'),
                    })
        except Exception as e:
            log.warning(f'list_knowledge: failed to fetch notes: {e}')

        return json.dumps(result)

    # ═══════════════════════════════════════════
    # Web Search Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_web(self, query: str, count: Optional[int] = None) -> str:
        """Web search. builtin.py: search_web(query, count=None)"""
        payload: dict[str, Any] = {'query': query}
        if count is not None:
            payload['max_results'] = count
        result = await self._post('/api/v1/retrieval/web/search', payload)
        if result:
            return json.dumps(result if isinstance(result, list) else [result])
        return json.dumps([])

    async def fetch_url(self, url: str) -> str:
        """Fetch URL content. builtin.py: fetch_url(url)"""
        result = await self._post('/api/v1/retrieval/web/loader', {'url': url})
        if result:
            content = result.get('content', '') if isinstance(result, dict) else str(result)
            return content
        return ''

    # ═══════════════════════════════════════════
    # Image Generation Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def generate_image(self, prompt: str) -> str:
        """Generate image. builtin.py: generate_image(prompt)"""
        result = await self._post('/api/v1/images/generations', {
            'prompt': prompt,
        })
        if result:
            return json.dumps({'status': 'success', 'images': result if isinstance(result, list) else [result]})
        return json.dumps({'error': 'Failed to generate image'})

    async def edit_image(self, prompt: str, image_urls: list[str]) -> str:
        """Edit image. builtin.py: edit_image(prompt, image_urls)"""
        result = await self._post('/api/v1/images/edits', {
            'prompt': prompt,
            'image': image_urls,
        })
        if result:
            return json.dumps({'status': 'success', 'images': result if isinstance(result, list) else [result]})
        return json.dumps({'error': 'Failed to edit image'})

    # ═══════════════════════════════════════════
    # Notes Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_notes(self, query: str, count: int = 5, start_timestamp: Optional[int] = None, end_timestamp: Optional[int] = None) -> str:
        """Search notes. builtin.py: search_notes(query, count=5, start_timestamp, end_timestamp)"""
        params: dict[str, Any] = {
            'query': query,
            'limit': count,
        }
        result = await self._get('/api/v1/notes/search', params=params)
        items = result.get('items', []) if result else []

        notes = []
        for n in items:
            note_data = n.get('data', {}) if n.get('data') else {}
            content_md = note_data.get('content', {}).get('md', '') if isinstance(note_data.get('content'), dict) else ''
            notes.append({
                'id': n.get('id'),
                'title': n.get('title'),
                'snippet': content_md[:200] if content_md else '',
                'updated_at': n.get('updated_at'),
            })
        return json.dumps(notes[:count])

    async def view_note(self, note_id: str) -> str:
        """View note. builtin.py: view_note(note_id)"""
        result = await self._get(f'/api/v1/notes/{note_id}')
        if result:
            note_data = result.get('data', {}) if result.get('data') else {}
            content_md = note_data.get('content', {}).get('md', '') if isinstance(note_data.get('content'), dict) else ''
            return json.dumps({
                'id': result.get('id'),
                'title': result.get('title'),
                'content': content_md,
                'updated_at': result.get('updated_at'),
                'created_at': result.get('created_at'),
            })
        return json.dumps({'error': 'Note not found'})

    async def write_note(self, title: str, content: str) -> str:
        """Create note. builtin.py: write_note(title, content)"""
        result = await self._post('/api/v1/notes', {
            'title': title,
            'data': {'content': {'md': content}},
        })
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'title': result.get('title')})
        return json.dumps({'error': 'Failed to create note'})

    async def replace_note_content(self, note_id: str, content: str, title: Optional[str] = None) -> str:
        """Update note content. builtin.py: replace_note_content(note_id, content, title)"""
        payload: dict[str, Any] = {'data': {'content': {'md': content}}}
        if title is not None:
            payload['title'] = title
        result = await self._put(f'/api/v1/notes/{note_id}', payload)
        if result:
            return json.dumps({
                'status': 'success',
                'id': result.get('id'),
                'title': result.get('title'),
                'updated_at': result.get('updated_at'),
            })
        return json.dumps({'error': 'Failed to update note'})

    # ═══════════════════════════════════════════
    # Chat Search Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_chats(self, query: str, count: int = 5, start_timestamp: Optional[int] = None, end_timestamp: Optional[int] = None) -> str:
        """Search chats. builtin.py: search_chats(query, count=5, start_timestamp, end_timestamp)"""
        result = await self._get(f'/api/v1/chats/search', params={
            'query': query,
            'limit': count * 3,  # Fetch more for filtering
        })
        items = result.get('items', []) if result else []

        chats = []
        for c in items:
            chats.append({
                'id': c.get('id'),
                'title': c.get('title'),
                'updated_at': c.get('updated_at'),
            })
        return json.dumps(chats[:count])

    async def view_chat(self, chat_id: str) -> str:
        """View chat. builtin.py: view_chat(chat_id)"""
        result = await self._get(f'/api/v1/chats/{chat_id}')
        if result:
            chat_data = result.get('chat', {}) if result.get('chat') else {}
            history = chat_data.get('history', {}) if isinstance(chat_data, dict) else {}
            messages = history.get('messages', {}) if isinstance(history, dict) else {}

            # Build message chain from currentId
            msg_list = []
            current_id = history.get('currentId') if isinstance(history, dict) else None
            visited = set()
            while current_id and current_id not in visited:
                visited.add(current_id)
                msg = messages.get(current_id) if isinstance(messages, dict) else None
                if msg:
                    msg_list.append({
                        'role': msg.get('role', ''),
                        'content': msg.get('content', ''),
                    })
                current_id = msg.get('parentId') if msg else None
            msg_list.reverse()

            return json.dumps({
                'id': result.get('id'),
                'title': result.get('title'),
                'messages': msg_list,
                'updated_at': result.get('updated_at'),
                'created_at': result.get('created_at'),
            })
        return json.dumps({'error': 'Chat not found'})

    # ═══════════════════════════════════════════
    # Skills Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def view_skill(self, id: str) -> str:
        """View skill. builtin.py: view_skill(id)"""
        result = await self._get(f'/api/v1/skills/{id}')
        if result:
            return json.dumps({
                'name': result.get('name'),
                'content': result.get('content', result.get('data', '')),
            })
        return json.dumps({'error': f'Skill {id} not found'})

    # ═══════════════════════════════════════════
    # Channel Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_channels(self, query: str, count: int = 5) -> str:
        """Search channels. builtin.py: search_channels(query, count=5)"""
        result = await self._get(f'/api/v1/channels', params={
            'query': query,
            'limit': count,
        })
        if isinstance(result, list):
            channels = result
        elif isinstance(result, dict) and 'items' in result:
            channels = result['items']
        else:
            channels = []

        return json.dumps([
            {
                'id': c.get('id'),
                'name': c.get('name'),
                'description': c.get('description', ''),
                'type': c.get('type', 'public'),
            }
            for c in channels[:count]
        ])

    async def search_channel_messages(self, query: str, count: int = 10, start_timestamp: Optional[int] = None, end_timestamp: Optional[int] = None) -> str:
        """
        Search channel messages. builtin.py: search_channel_messages(query, count=10, start_timestamp, end_timestamp)
        NOTE: Channel message search requires WebSocket context unavailable via proxy API.
        """
        return json.dumps({
            'error': 'search_channel_messages is not available through the proxy API. '
                     'Channel message search requires real-time WebSocket connection.'
        })

    async def view_channel_message(self, message_id: str) -> str:
        """
        View a channel message. builtin.py: view_channel_message(message_id)
        """
        result = await self._get(f'/api/v1/channels/messages/{message_id}')
        if result:
            return json.dumps({
                'id': result.get('id'),
                'channel_id': result.get('channel_id'),
                'channel_name': result.get('channel_name'),
                'content': result.get('content'),
                'user_id': result.get('user_id'),
                'created_at': result.get('created_at'),
            })
        return json.dumps({'error': 'Message not found'})

    async def view_channel_thread(self, parent_message_id: str) -> str:
        """
        View a channel thread. builtin.py: view_channel_thread(parent_message_id)
        """
        result = await self._get(f'/api/v1/channels/messages/{parent_message_id}/thread')
        if result:
            return json.dumps({
                'thread_id': parent_message_id,
                'messages': result.get('messages', []),
                'message_count': len(result.get('messages', [])),
            })
        return json.dumps({'error': 'Thread not found'})

    # ═══════════════════════════════════════════
    # Calendar Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def search_calendar_events(self, query: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, count: int = 10) -> str:
        """Search calendar events. builtin.py: search_calendar_events(query, start, end, count=10)"""
        params: dict[str, Any] = {}
        if query:
            params['query'] = query
        if start:
            params['start'] = start
        if end:
            params['end'] = end
        params['limit'] = count
        result = await self._get('/api/v1/calendars/events', params)
        if isinstance(result, dict):
            items = result.get('events', result.get('items', []))
        else:
            items = result or []
        return json.dumps(items[:count] if items else [])

    async def create_calendar_event(self, title: str, start: str, end: Optional[str] = None, description: Optional[str] = None, calendar_id: Optional[str] = None, all_day: bool = False, location: Optional[str] = None, reminder_minutes: Optional[int] = None) -> str:
        """Create calendar event. builtin.py: create_calendar_event(title, start, end, description, calendar_id, all_day, location, reminder_minutes)"""
        payload: dict[str, Any] = {
            'title': title,
            'start': start,
        }
        if end:
            payload['end'] = end
        if description:
            payload['description'] = description
        if calendar_id:
            payload['calendar_id'] = calendar_id
        payload['all_day'] = all_day
        if location:
            payload['location'] = location
        if reminder_minutes is not None:
            payload['reminder_minutes'] = reminder_minutes
        result = await self._post('/api/v1/calendars/events', payload)
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'title': title})
        return json.dumps({'error': 'Failed to create event'})

    async def update_calendar_event(self, event_id: str, title: Optional[str] = None, description: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, all_day: Optional[bool] = None, location: Optional[str] = None, is_cancelled: Optional[bool] = None, reminder_minutes: Optional[int] = None) -> str:
        """Update calendar event. builtin.py: update_calendar_event(event_id, title, description, start, end, all_day, location, is_cancelled, reminder_minutes)"""
        payload: dict[str, Any] = {}
        if title is not None:
            payload['title'] = title
        if description is not None:
            payload['description'] = description
        if start is not None:
            payload['start'] = start
        if end is not None:
            payload['end'] = end
        if all_day is not None:
            payload['all_day'] = all_day
        if location is not None:
            payload['location'] = location
        if is_cancelled is not None:
            payload['is_cancelled'] = is_cancelled
        if reminder_minutes is not None:
            payload['reminder_minutes'] = reminder_minutes
        result = await self._put(f'/api/v1/calendars/events/{event_id}', payload)
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id')})
        return json.dumps({'error': 'Failed to update event'})

    async def delete_calendar_event(self, event_id: str) -> str:
        """Delete calendar event. builtin.py: delete_calendar_event(event_id)"""
        result = await self._delete(f'/api/v1/calendars/events/{event_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})

    # ═══════════════════════════════════════════
    # Automation Tools  (signatures match builtin.py)
    # ═══════════════════════════════════════════

    async def create_automation(self, name: str, prompt: str, rrule: str) -> str:
        """Create automation. builtin.py: create_automation(name, prompt, rrule)"""
        result = await self._post('/api/v1/automations', {
            'name': name,
            'data': {'prompt': prompt, 'rrule': rrule},
            'is_active': True,
        })
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id'), 'name': name})
        return json.dumps({'error': 'Failed to create automation'})

    async def update_automation(self, automation_id: str, name: Optional[str] = None, prompt: Optional[str] = None, rrule: Optional[str] = None, model_id: Optional[str] = None) -> str:
        """Update automation. builtin.py: update_automation(automation_id, name, prompt, rrule, model_id)"""
        payload: dict[str, Any] = {}
        if name is not None:
            payload['name'] = name
        data: dict[str, Any] = {}
        if prompt is not None:
            data['prompt'] = prompt
        if rrule is not None:
            data['rrule'] = rrule
        if model_id is not None:
            data['model_id'] = model_id
        if data:
            payload['data'] = data
        result = await self._put(f'/api/v1/automations/{automation_id}', payload)
        if result:
            return json.dumps({'status': 'success', 'id': result.get('id')})
        return json.dumps({'error': 'Failed to update automation'})

    async def list_automations(self, status: Optional[str] = None, count: int = 10) -> str:
        """List automations. builtin.py: list_automations(status, count=10)"""
        params: dict[str, Any] = {'limit': count}
        if status:
            params['status'] = status
        result = await self._get('/api/v1/automations', params=params)
        if isinstance(result, dict):
            items = result.get('automations', result.get('items', []))
        else:
            items = result or []
        return json.dumps([
            {
                'id': a.get('id'),
                'name': a.get('name'),
                'is_active': a.get('is_active', True),
                'prompt_snippet': a.get('prompt_snippet', ''),
                'next_runs': a.get('next_runs', []),
            }
            for a in items[:count]
        ])

    async def toggle_automation(self, automation_id: str) -> str:
        """Toggle automation. builtin.py: toggle_automation(automation_id)"""
        result = await self._post(f'/api/v1/automations/{automation_id}/toggle')
        return json.dumps({'status': 'success', 'id': automation_id})

    async def delete_automation(self, automation_id: str) -> str:
        """Delete automation. builtin.py: delete_automation(automation_id)"""
        result = await self._delete(f'/api/v1/automations/{automation_id}')
        return json.dumps({'status': 'success', 'deleted': bool(result)})


# Singleton client instance
client = OpenWebUIClient()