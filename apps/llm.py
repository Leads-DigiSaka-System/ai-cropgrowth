"""LLM backends for the agent: Gemini (google-genai) and any OpenAI-compatible chat API (OpenRouter → Qwen).

The agent keeps its conversation in one provider-neutral form (OpenAI chat format: user / assistant with
tool_calls / tool messages), so a conversation can move from Gemini to Qwen mid-way when Gemini is busy.
Each backend's complete() returns {"text": str, "tool_calls": [{"id", "name", "args", ...}], "message": dict},
where "message" is the assistant turn to append to that history.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid

DEFAULT_GEMINI = "gemini-3.1-flash"
DEFAULT_QWEN = "qwen/qwen3-235b-a22b-2507"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

TRANSIENT_CODES = {408, 429, 500, 502, 503, 504, 529}


class TransientLLMError(RuntimeError):
    """Rate limit, overload or network trouble: worth a retry, or a switch to the fallback model."""


# ---------------------------------------------------------------- Gemini
class GeminiBackend:
    def __init__(self, model: str = DEFAULT_GEMINI, api_key: str | None = None):
        from google import genai
        self.model, self.name = model, f"gemini:{model}"
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

    def _contents(self, messages: list[dict]):
        from google.genai import types
        out, pending = [], []                       # consecutive tool results → one user Content
        for m in messages:
            if m["role"] == "tool":
                pending.append(types.Part(function_response=types.FunctionResponse(
                    name=m["name"], response={"result": json.loads(m["content"])}, id=m.get("_gemini_id"))))
                continue
            if pending:
                out.append(types.Content(role="user", parts=pending)); pending = []
            if m["role"] == "user":
                out.append(types.Content(role="user", parts=[types.Part(text=m["content"])]))
            elif "_gemini" in m:                    # Gemini's own turn: resend as-is (keeps thought signatures)
                out.append(m["_gemini"])
            else:                                   # a turn produced by the fallback model
                parts = [types.Part(text=m["content"])] if m.get("content") else []
                parts += [types.Part(function_call=types.FunctionCall(
                    name=tc["function"]["name"], args=json.loads(tc["function"]["arguments"] or "{}")))
                    for tc in m.get("tool_calls") or []]
                out.append(types.Content(role="model", parts=parts or [types.Part(text="")]))
        if pending:
            out.append(types.Content(role="user", parts=pending))
        return out

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4000) -> dict:
        from google.genai import types
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                             tools=[types.Tool(function_declarations=tools)])
        try:
            resp = self.client.models.generate_content(model=self.model, contents=self._contents(messages),
                                                       config=config)
        except Exception as e:
            if getattr(e, "code", None) in TRANSIENT_CODES or isinstance(e, (ConnectionError, TimeoutError)):
                raise TransientLLMError(f"{self.name}: {e}") from e
            raise
        cand = resp.candidates[0] if resp.candidates else None
        if cand is None or cand.content is None or not cand.content.parts:
            reason = getattr(cand, "finish_reason", None) if cand else getattr(resp, "prompt_feedback", None)
            return {"text": f"Gemini returned no content (finish reason: {reason}).", "tool_calls": [],
                    "message": None}
        parts = cand.content.parts
        calls = [{"id": f"call_{uuid.uuid4().hex[:16]}", "gemini_id": getattr(p.function_call, "id", None),
                  "name": p.function_call.name, "args": dict(p.function_call.args or {})}
                 for p in parts if p.function_call]
        text = "".join(p.text for p in parts if p.text)
        message = {"role": "assistant", "content": text or None, "_gemini": cand.content,
                   "tool_calls": [{"id": c["id"], "type": "function",
                                   "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                                  for c in calls] or None}
        return {"text": text, "tool_calls": calls, "message": message}


# ---------------------------------------------------------------- OpenAI-compatible (OpenRouter)
class OpenAICompatBackend:
    """Chat Completions API with tool calling. Defaults to OpenRouter + Qwen; works with any compatible server."""

    def __init__(self, model: str = DEFAULT_QWEN, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 180):
        self.model, self.name = model, f"openrouter:{model}"
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or OPENROUTER_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("no OpenRouter API key: set OPENROUTER_API_KEY")
        self.timeout = timeout

    @staticmethod
    def _clean(m: dict) -> dict:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m.get("tool_calls") is None:
            m.pop("tool_calls", None)
        if m["role"] == "tool":
            m.pop("name", None)
        return m

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "X-Title": "OpERA field-processing agent"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            if e.code in TRANSIENT_CODES:
                raise TransientLLMError(f"{self.name}: HTTP {e.code} {body}") from e
            raise RuntimeError(f"{self.name}: HTTP {e.code} {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise TransientLLMError(f"{self.name}: {e}") from e

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 4000) -> dict:
        data = self._post({"model": self.model, "max_tokens": max_tokens,
                           "messages": [{"role": "system", "content": system}] + [self._clean(m) for m in messages],
                           "tools": [{"type": "function", "function": t} for t in tools]})
        if "error" in data:                         # OpenRouter reports upstream failures in the body
            err = data["error"]
            code = err.get("code") if isinstance(err, dict) else None
            exc = TransientLLMError if code in TRANSIENT_CODES else RuntimeError
            raise exc(f"{self.name}: {err}")
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc["id"], "name": tc["function"]["name"], "args": args})
        message = {"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls") or None}
        return {"text": msg.get("content") or "", "tool_calls": calls, "message": message}


def make_backend(provider: str, model: str | None = None, api_key: str | None = None):
    if provider == "gemini":
        return GeminiBackend(model or DEFAULT_GEMINI, api_key)
    if provider == "openrouter":
        return OpenAICompatBackend(model or os.environ.get("OPENROUTER_MODEL") or DEFAULT_QWEN, api_key)
    raise ValueError(f"unknown provider {provider!r} (gemini | openrouter)")
