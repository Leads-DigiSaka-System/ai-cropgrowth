import json

import pytest

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent.agent import CropGrowthAgent
from cropgrowth_agent.llm import ModelUnavailableError, OpenAICompatBackend, TransientLLMError
from cropgrowth_agent.runner import RunConfig, Runner
from cropgrowth_agent.tools import TOOLS


class Scripted:
    """Fake backend: raises `fail` errors first, then plays back scripted replies."""
    def __init__(self, name, replies=(), fail=0, exc=TransientLLMError):
        self.name, self.model, self.replies, self.fail, self.exc, self.seen = name, name, list(replies), fail, exc, []

    def complete(self, system, messages, tools, max_tokens=4000):
        self.seen.append([dict(m) for m in messages])
        if self.fail:
            self.fail -= 1
            raise self.exc(f"{self.name}: 429 RESOURCE_EXHAUSTED")
        return self.replies.pop(0)


def call(name, args, i="c1"):
    msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
    return {"text": "", "tool_calls": [{"id": i, "name": name, "args": args}], "message": msg}


def answer(text):
    return {"text": text, "tool_calls": [], "message": {"role": "assistant", "content": text}}


@pytest.fixture
def runner(mpc, boundaries, tmp_path):
    cfg = RunConfig(resolution_m=dp.GRID_SCALE_DEG * dp.M_PER_DEG, tile_deg=0.1, tile_workers=1,
                    date_median_radius=0, vector_path=boundaries["provinces"], region_col="Reg_Name",
                    aoi_path=boundaries["aoi"], output_target="local", local_root=str(tmp_path / "out"),
                    season_year=2026)
    return Runner(cfg, log=lambda *a: None)


def agent(runner, primary, fallback=None, **kw):
    kw.setdefault("confirm", None)
    a = CropGrowthAgent(runner=runner, backend=primary, fallback_backends=[fallback] if fallback else [],
                        verbose=False, **kw)
    a._sleep = lambda s: None
    return a


def test_tool_loop(runner):
    g = Scripted("gemini", [call("list_areas", {"kind": "aoi", "contains": "muñoz"}), answer("found it")])
    a = agent(runner, g)
    assert a.run("find Muñoz") == "found it"
    tool_msg = g.seen[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["names"] == ["MUÑOZ TOWN"]


def test_recent_request_end_to_end(runner):
    g = Scripted("gemini", [call("update_settings", {"output_target": "local", "season_year": 2026.0}, "c1"),
                            call("run_recent", {"names": ["MUÑOZ TOWN"], "as_of": "2026-03-20"}, "c2"),
                            answer("mostly vegetative")])
    a = agent(runner, g)
    assert a.run("current stage of Muñoz") == "mostly vegetative"
    assert runner.cfg.season_year == 2026 and isinstance(runner.cfg.season_year, int)   # Gemini floats coerced
    res = json.loads(a.messages[-2]["content"])
    assert list(res["success"][0]["hectares"]) == ["vegetative"]
    assert "_data" not in res["success"][0]                                    # no arrays sent to the model


def test_retries_then_succeeds_on_primary(runner):
    g = Scripted("gemini", [answer("ok")], fail=2)
    q = Scripted("qwen", [answer("from qwen")])
    assert agent(runner, g, q, retries=2).run("hi") == "ok"
    assert q.seen == []


class BusyAfterFirst(Scripted):
    def complete(self, system, messages, tools, max_tokens=4000):
        if self.seen:
            self.seen.append(None)
            raise TransientLLMError(f"{self.name}: 503 UNAVAILABLE")
        return super().complete(system, messages, tools, max_tokens)


def test_falls_back_to_qwen_mid_conversation(runner):
    g = BusyAfterFirst("gemini", [call("get_settings", {})])
    q = Scripted("qwen", [answer("done on qwen")])
    assert agent(runner, g, q, retries=1).run("settings?") == "done on qwen"
    assert len(g.seen) == 3
    assert [m["role"] for m in q.seen[0]] == ["user", "assistant", "tool"]


def test_unavailable_model_skips_retries(runner):
    g = Scripted("gemini-old", [], fail=1, exc=ModelUnavailableError)
    q = Scripted("qwen", [answer("ok")])
    assert agent(runner, g, q, retries=3).run("hi") == "ok" and len(g.seen) == 1


def test_next_request_tries_primary_again(runner):
    g = Scripted("gemini", [answer("gemini back")], fail=3)
    q = Scripted("qwen", [answer("qwen 1")])
    a = agent(runner, g, q, retries=2)
    assert a.run("first") == "qwen 1"
    assert a.run("second") == "gemini back"


def test_periodic_run_needs_approval(runner):
    asked = []
    g = Scripted("gemini", [call("run_periodic", {"level": "regional", "names": ["R1"], "as_of": "2026-05-01"}),
                            answer("not run")])
    a = agent(runner, g, confirm=lambda t, args, plan: asked.append(plan) or False)
    a.run("make April maps for R1")
    assert json.loads(a.messages[-2]["content"])["status"] == "cancelled"
    assert asked[0]["provinces"] == 2
    assert not any("202604" in f for f in runner.list_outputs("periodic")["files"])


def test_agent_cannot_change_paths(runner):
    a = agent(runner, Scripted("gemini"))
    r = a.tools("update_settings", {"local_root": "/etc", "cadence": "semimonthly"})
    assert "error" in r and runner.cfg.local_root != "/etc" and runner.cfg.cadence == "monthly"


def test_method_docs_search(runner):
    a = agent(runner, Scripted("gemini"))
    hits = a.tools("search_method_docs", {"query": "heading date NDVI_norm", "k": 3})["results"]
    assert hits and any("phenology.py" in h["source"] for h in hits)


def test_backends_serialise_the_history(runner, monkeypatch):
    g = Scripted("gemini", [call("list_areas", {"kind": "province"}), answer("3 provinces")])
    a = agent(runner, g)
    a.run("list provinces")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    oa, sent = OpenAICompatBackend(), {}
    oa._post = lambda payload: sent.setdefault("p", payload) and {"choices": [{"message": {"content": "ok"}}]}
    assert oa.complete("sys", a.messages, TOOLS)["text"] == "ok"
    tool = next(m for m in sent["p"]["messages"] if m["role"] == "tool")
    assert "tool_call_id" in tool and "name" not in tool and "_gemini_id" not in tool

    pytest.importorskip("google.genai")
    from google.genai import types
    from cropgrowth_agent.llm import GeminiBackend
    types.Tool(function_declarations=TOOLS)                     # schemas accepted by the Gemini SDK
    contents = GeminiBackend(api_key="dummy")._contents(a.messages)
    assert any(p.function_response for c in contents for p in c.parts)
