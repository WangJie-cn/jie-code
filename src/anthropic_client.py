"""Anthropic Messages API client for jie-code.

Implements the Anthropic Messages API (POST /v1/messages) with SSE streaming,
returning the same AssistantTurn/StreamEvent types as OpenAICompatClient.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Iterator
from urllib import error, request

from .agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
    ToolCall,
    UsageStats,
)


class AnthropicClientError(RuntimeError):
    """Raised when the Anthropic API returns an error."""


def _join_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip('/')
    return f'{base}/{suffix.lstrip("/")}'


def _optional_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _convert_tools_openai_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling tool format to Anthropic tool format."""
    result = []
    for tool in tools:
        if tool.get('type') == 'function':
            func = tool.get('function', {})
            result.append({
                'name': func.get('name', ''),
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {'type': 'object', 'properties': {}}),
            })
        elif 'name' in tool and 'input_schema' in tool:
            result.append(tool)
        elif 'name' in tool:
            result.append({
                'name': tool['name'],
                'description': tool.get('description', ''),
                'input_schema': tool.get('parameters', {'type': 'object', 'properties': {}}),
            })
    return result


def _convert_messages_openai_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI messages to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    Anthropic requires system prompt to be separate, not in messages.
    """
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'system':
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        system_parts.append(item)
                    elif isinstance(item, dict) and item.get('type') == 'text':
                        system_parts.append(item.get('text', ''))
            continue

        if role == 'tool':
            # OpenAI tool result -> Anthropic tool_result content block
            anthropic_messages.append({
                'role': 'user',
                'content': [{
                    'type': 'tool_result',
                    'tool_use_id': msg.get('tool_call_id', ''),
                    'content': content if isinstance(content, str) else json.dumps(content),
                }],
            })
            continue

        if role == 'assistant':
            # Check for tool_calls in assistant message
            tool_calls = msg.get('tool_calls', [])
            if tool_calls:
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({'type': 'text', 'text': content})
                for tc in tool_calls:
                    func = tc.get('function', {})
                    args = func.get('arguments', '{}')
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    blocks.append({
                        'type': 'tool_use',
                        'id': tc.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                        'name': func.get('name', ''),
                        'input': args,
                    })
                anthropic_messages.append({'role': 'assistant', 'content': blocks})
            else:
                anthropic_messages.append({'role': 'assistant', 'content': content or ''})
            continue

        # user messages
        anthropic_messages.append({'role': 'user', 'content': content or ''})

    # Anthropic requires alternating user/assistant messages.
    # Merge consecutive same-role messages if needed.
    merged: list[dict[str, Any]] = []
    for msg in anthropic_messages:
        if merged and merged[-1]['role'] == msg['role']:
            prev = merged[-1]['content']
            curr = msg['content']
            # Normalize to lists for merging
            if isinstance(prev, str):
                prev = [{'type': 'text', 'text': prev}] if prev else []
            if isinstance(curr, str):
                curr = [{'type': 'text', 'text': curr}] if curr else []
            merged[-1]['content'] = prev + curr
        else:
            merged.append(msg)

    return '\n\n'.join(system_parts), merged


class AnthropicClient:
    """Anthropic Messages API client."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {
            'Content-Type': 'application/json',
            'anthropic-version': '2023-06-01',
        }
        key = self.config.api_key
        if key.startswith('sk-ant-'):
            headers['x-api-key'] = key
        else:
            headers['Authorization'] = f'Bearer {key}'
        return headers

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn:
        payload = self._build_payload(messages, tools, stream=False)
        response_data = self._request_json(payload)
        return self._parse_response(response_data)

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]:
        payload = self._build_payload(messages, tools, stream=True)
        url = _join_url(self.config.base_url, '/v1/messages')
        req = request.Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=self._headers(),
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                yield StreamEvent(type='message_start')
                yield from self._parse_sse_stream(response)
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise AnthropicClientError(
                f'HTTP {exc.code} from Anthropic API: {detail}'
            ) from exc
        except error.URLError as exc:
            raise AnthropicClientError(
                f'Unable to reach Anthropic API at {self.config.base_url}: {exc.reason}'
            ) from exc

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        system_prompt, anthropic_messages = _convert_messages_openai_to_anthropic(messages)
        anthropic_tools = _convert_tools_openai_to_anthropic(tools)

        payload: dict[str, Any] = {
            'model': self.config.model,
            'max_tokens': 16384,
            'messages': anthropic_messages,
            'stream': stream,
        }
        if system_prompt:
            payload['system'] = system_prompt
        if anthropic_tools:
            payload['tools'] = anthropic_tools
        if self.config.temperature > 0:
            payload['temperature'] = self.config.temperature
        return payload

    def _request_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = _join_url(self.config.base_url, '/v1/messages')
        body = json.dumps(payload).encode('utf-8')
        req = request.Request(url, data=body, headers=self._headers(), method='POST')
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')
            raise AnthropicClientError(
                f'HTTP {exc.code} from Anthropic API: {detail}'
            ) from exc
        except error.URLError as exc:
            raise AnthropicClientError(
                f'Unable to reach Anthropic API at {self.config.base_url}: {exc.reason}'
            ) from exc
        try:
            data = json.loads(raw.decode('utf-8'))
        except json.JSONDecodeError as exc:
            raise AnthropicClientError('Anthropic API returned invalid JSON') from exc
        if not isinstance(data, dict):
            raise AnthropicClientError('Anthropic API returned malformed payload')
        if data.get('type') == 'error':
            err = data.get('error', {})
            raise AnthropicClientError(
                f"Anthropic API error: {err.get('type', 'unknown')}: {err.get('message', '')}"
            )
        return data

    def _parse_response(self, data: dict[str, Any]) -> AssistantTurn:
        """Parse a non-streaming Anthropic Messages API response."""
        content_blocks = data.get('content', [])
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get('type')
            if block_type == 'text':
                text_parts.append(block.get('text', ''))
            elif block_type == 'tool_use':
                tool_calls.append(ToolCall(
                    id=block.get('id', f'toolu_{uuid.uuid4().hex[:24]}'),
                    name=block.get('name', ''),
                    arguments=block.get('input', {}),
                ))

        usage_data = data.get('usage', {})
        usage = UsageStats(
            input_tokens=_optional_int(usage_data.get('input_tokens')),
            output_tokens=_optional_int(usage_data.get('output_tokens')),
            cache_creation_input_tokens=_optional_int(usage_data.get('cache_creation_input_tokens')),
            cache_read_input_tokens=_optional_int(usage_data.get('cache_read_input_tokens')),
        )

        return AssistantTurn(
            content=''.join(text_parts),
            tool_calls=tuple(tool_calls),
            finish_reason=data.get('stop_reason', 'end_turn'),
            raw_message=data,
            usage=usage,
        )

    def _parse_sse_stream(self, response: Any) -> Iterator[StreamEvent]:
        """Parse Anthropic SSE stream events."""
        current_tool_index = -1
        tool_id_map: dict[int, str] = {}

        for event_type, event_data in self._iter_sse_events(response):
            if event_type == 'message_start':
                msg = event_data.get('message', {})
                usage_data = msg.get('usage', {})
                if usage_data:
                    yield StreamEvent(
                        type='usage',
                        usage=UsageStats(
                            input_tokens=_optional_int(usage_data.get('input_tokens')),
                        ),
                        raw_event=event_data,
                    )

            elif event_type == 'content_block_start':
                block = event_data.get('content_block', {})
                if block.get('type') == 'tool_use':
                    current_tool_index += 1
                    tool_id = block.get('id', f'toolu_{uuid.uuid4().hex[:24]}')
                    tool_id_map[current_tool_index] = tool_id
                    yield StreamEvent(
                        type='tool_call_delta',
                        tool_call_index=current_tool_index,
                        tool_call_id=tool_id,
                        tool_name=block.get('name'),
                        arguments_delta='',
                        raw_event=event_data,
                    )

            elif event_type == 'content_block_delta':
                delta = event_data.get('delta', {})
                delta_type = delta.get('type')
                if delta_type == 'text_delta':
                    yield StreamEvent(
                        type='content_delta',
                        delta=delta.get('text', ''),
                        raw_event=event_data,
                    )
                elif delta_type == 'input_json_delta':
                    yield StreamEvent(
                        type='tool_call_delta',
                        tool_call_index=current_tool_index,
                        tool_call_id=tool_id_map.get(current_tool_index),
                        arguments_delta=delta.get('partial_json', ''),
                        raw_event=event_data,
                    )

            elif event_type == 'message_delta':
                delta = event_data.get('delta', {})
                usage_data = event_data.get('usage', {})
                stop_reason = delta.get('stop_reason')
                if usage_data:
                    yield StreamEvent(
                        type='usage',
                        usage=UsageStats(
                            output_tokens=_optional_int(usage_data.get('output_tokens')),
                        ),
                        raw_event=event_data,
                    )
                if stop_reason:
                    yield StreamEvent(
                        type='message_stop',
                        finish_reason=stop_reason,
                        raw_event=event_data,
                    )

            elif event_type == 'message_stop':
                pass  # Already handled via message_delta stop_reason

            elif event_type == 'error':
                err = event_data.get('error', {})
                raise AnthropicClientError(
                    f"Stream error: {err.get('type', 'unknown')}: {err.get('message', '')}"
                )

    def _iter_sse_events(self, response: Any) -> Iterator[tuple[str, dict[str, Any]]]:
        """Parse raw SSE lines into (event_type, data_dict) tuples."""
        event_type = ''
        data_lines: list[str] = []

        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                text = line.decode('utf-8', errors='replace')
            else:
                text = str(line)
            stripped = text.rstrip('\r\n')

            if not stripped:
                # Empty line = end of event
                if data_lines:
                    joined = '\n'.join(data_lines)
                    try:
                        data = json.loads(joined)
                    except json.JSONDecodeError:
                        data = {}
                    if event_type:
                        yield event_type, data if isinstance(data, dict) else {}
                event_type = ''
                data_lines.clear()
                continue

            if stripped.startswith('event:'):
                event_type = stripped[6:].strip()
            elif stripped.startswith('data:'):
                data_lines.append(stripped[5:].strip())

        # Flush remaining
        if data_lines and event_type:
            joined = '\n'.join(data_lines)
            try:
                data = json.loads(joined)
            except json.JSONDecodeError:
                data = {}
            yield event_type, data if isinstance(data, dict) else {}
