"""
agent.py
==================================================================
CropGrowth Agent: a tool-using LLM agent that runs the rice growth-stage
pipeline from plain-language requests, e.g.

    "Map the current growth stage of Science City of Muñoz and save it to Drive"
    "Run the February maps for Region III, semimonthly"
    "Which provinces failed last run? Retry them."

How it works
  * The LLM (Gemini by default, Qwen through OpenRouter as fallback — see
    llm.py) plans and calls TOOLS; the tools are thin wrappers around
    runner.Runner, so every number it reports comes from the pipeline.
  * Gemini rate limits / overload / network errors are retried, then the
    conversation moves to the fallback model mid-turn (llm.py keeps one
    provider-neutral history).
  * Expensive tools (run_periodic by default) need human approval: the
    agent shows the plan (provinces, tiles, periods, output location) and
    calls `confirm(tool, args, plan) -> bool` first.

    agent = CropGrowthAgent(runner)                  # GEMINI_API_KEY / OPENROUTER_API_KEY
    print(agent.chat("What areas can you map?"))
==================================================================
"""

from __future__ import annotations

import json
import time

import pandas as pd

import phenology as ph
import pipeline as pl
from llm import TransientLLMError, make_backend

# ------------------------------------------------------------------ tool schemas
_STR = {"type": "string"}
_STRS = {"type": "array", "items": {"type": "string"}}
_BBOX = {"type": "array", "items": {"type": "number"},
         "description": "west, south, east, north in degrees (EPSG:4326)"}
_DATE = {"type": "string", "description": "YYYY-MM-DD; omit for today"}

TOOLS = [
    {"name": "get_settings",                       # no parameters: Gemini rejects empty objects
     "description": "Current pipeline settings: data source, grid, cadence, periods, output "
                    "destination, phenology parameters, boundary files."},
    {"name": "update_settings",
     "description": "Change pipeline settings. Only pass the fields to change. output_target "
                    "chooses where products are saved: 'gcs' (Cloud Storage), 'gdrive' (Google "
                    "Drive), 'both', or 'local'.",
     "parameters": {"type": "object", "properties": {
         "data_source": {"type": "string", "enum": ["hls", "s2"],
                         "description": "hls = Landsat+Sentinel-2 harmonized 30 m; s2 = Sentinel-2"},
         "resolution_m": {"type": "number", "description": "30 (default) or 10 (s2 only)"},
         "cadence": {"type": "string", "enum": list(pl.CADENCES)},
         "periods": {"type": "string", "enum": ["last_complete", "current", "season"]},
         "stage_method": {"type": "string", "enum": ["dominant", "midpoint"]},
         "output_target": {"type": "string", "enum": ["gcs", "gdrive", "both", "local"]},
         "year": {"type": "integer", "description": "season year (planting month of this year)"},
         "lookback_days": {"type": "integer", "description": "recent-mode window length"},
         "tile_workers": {"type": "integer"},
         "pheno": {"type": "object", "description": "phenology thresholds to change",
                   "properties": {
                       "THRESHOLD_FRAC": {"type": "number"}, "MIN_AMPLITUDE": {"type": "number"},
                       "MIN_PEAK_NDVI": {"type": "number"}, "MAX_BASE_NDVI": {"type": "number"},
                       "HEADING_NORM": {"type": "number"}, "SG_WINDOW": {"type": "integer"}}}}}},
    {"name": "list_areas",
     "description": "Names that can be mapped. kind='province' or 'region' (province boundary "
                    "file) or 'aoi' (municipal / barangay file). Use to resolve the exact "
                    "spelling of a place before running.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["province", "region", "aoi"]},
         "region": {"type": "string", "description": "only provinces of this region"},
         "contains": {"type": "string", "description": "case-insensitive substring filter"}},
         "required": ["kind"]}},
    {"name": "plan_periodic_run",
     "description": "Dry run for national / regional / provincial maps: which provinces, how "
                    "many tiles, which periods, where outputs go. Downloads nothing.",
     "parameters": {"type": "object", "properties": {
         "level": {"type": "string", "enum": list(pl.LEVELS)},
         "names": {**_STRS, "description": "regions (level=regional) or provinces (provincial)"},
         "as_of": _DATE}, "required": ["level"]}},
    {"name": "run_periodic",
     "description": "Produce the periodic growth-stage maps (one per province per period) for "
                    "national / regional / provincial areas and save them. Long-running; the "
                    "user is asked to approve first.",
     "parameters": {"type": "object", "properties": {
         "level": {"type": "string", "enum": list(pl.LEVELS)},
         "names": _STRS, "as_of": _DATE,
         "skip_existing": {"type": "boolean", "description": "skip maps already saved (default true)"},
         "only": {**_STRS, "description": "restrict to these provinces, e.g. to retry failures"}},
         "required": ["level"]}},
    {"name": "run_recent",
     "description": "Current (most recent) growth stage for municipal / barangay areas from the "
                    "AOI boundary file, or for a bbox. Saves a stage map + hectares per stage.",
     "parameters": {"type": "object", "properties": {
         "names": {**_STRS, "description": "names from list_areas(kind='aoi')"},
         "bbox": _BBOX, "as_of": _DATE,
         "lookback_days": {"type": "integer"}}}},
    {"name": "quick_check",
     "description": "Run phenology on a small bbox (a few km) and return QC outcome counts — "
                    "use to sanity-check thresholds before a large run.",
     "parameters": {"type": "object", "properties": {
         "bbox": _BBOX, "planting_month": {"type": "integer"}, "year": {"type": "integer"}},
         "required": ["bbox"]}},
    {"name": "list_outputs",
     "description": "Files already saved in the output destination.",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["periodic", "recent"]},
         "contains": {"type": "string"}}, "required": ["kind"]}},
    {"name": "read_summary",
     "description": "Hectares per growth stage from a saved *_summary.csv / summary/*.csv path "
                    "(as listed by list_outputs).",
     "parameters": {"type": "object", "properties": {"path": _STR}, "required": ["path"]}},
    {"name": "last_run",
     "description": "Result of the last periodic or recent run in this session (successes, "
                    "failures with errors, outputs).",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["periodic", "recent"]}}, "required": ["kind"]}},
]

SYSTEM_PROMPT = """You are CropGrowth Agent. You map rice growth stages in the Philippines from
satellite NDVI (Harmonized Landsat Sentinel-2 or Sentinel-2) by calling tools. Today is {today}.

Products
- Periodic mode (national / regional / provincial areas): one stage map per province per period
  (monthly, or semimonthly = 1st-15th and 16th-end), season-based. Tools: plan_periodic_run, run_periodic.
- Recent mode (municipal / barangay areas): the CURRENT growth stage plus how many days old the
  newest cloud-free observation is. Tool: run_recent.
- Stage classes: {stages}. -1 = not cropland.
- QC codes: {qc}.

How to work
1. Pick the mode from the area: municipality / barangay / city / small bbox -> run_recent;
   province / region / whole country -> periodic.
2. Resolve place names with list_areas before running; if a name is ambiguous or missing, ask.
3. Before run_periodic, call plan_periodic_run and state the plan (provinces, tiles, periods,
   destination). run_periodic asks the user for approval itself.
4. When the user says where to save (Google Drive, Cloud Storage, both), call update_settings
   with output_target first.
5. Report results only from tool outputs: what was made, where it was saved (URIs), hectares or
   shares per stage, data age, and every failure with its error. Never invent numbers.
6. For failed provinces caused by network / catalog errors, offer to retry them with
   run_periodic(only=[...]). If a tool returns an error, read it and fix the call or explain.
7. Be concise. Use plain language; the users are agriculture staff.
"""


_SCHEMAS = {t["name"]: t.get("parameters", {}) for t in TOOLS}


def _coerce(schema, value):
    """Cast tool arguments to their declared types: Gemini returns every
    number as a float (year 2025.0), and some models send numbers as strings."""
    t = (schema or {}).get("type")
    try:
        if t == "object" and isinstance(value, dict):
            props = schema.get("properties", {})
            return {k: _coerce(props.get(k), v) for k, v in value.items()}
        if t == "array" and isinstance(value, (list, tuple)):
            return [_coerce(schema.get("items"), v) for v in value]
        if t == "integer" and value is not None and not isinstance(value, bool):
            return int(float(value))
        if t == "number" and isinstance(value, str):
            return float(value)
        if t == "boolean" and isinstance(value, str):
            return value.strip().lower() in ("true", "yes", "1")
    except (TypeError, ValueError):
        pass
    return value


def console_confirm(tool, args, plan):
    """Default approval prompt (notebook / terminal)."""
    print(f"\n[approval needed] {tool}({json.dumps(args, default=str)})")
    if plan:
        print(json.dumps({k: v for k, v in plan.items() if k != "detail"}, indent=1, default=str))
    return input("Proceed? [y/N] ").strip().lower() in ("y", "yes")


def _public(obj):
    """Drop private keys (e.g. '_data' xarray payloads) for the LLM."""
    if isinstance(obj, dict):
        return {k: _public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [_public(v) for v in obj]
    return obj


def _clip_json(obj, limit=12000):
    s = json.dumps(_public(obj), default=str)
    if len(s) <= limit:
        return s
    return json.dumps({"truncated": True, "preview": s[:limit]})


class CropGrowthAgent:
    def __init__(self, runner, primary="gemini", fallback="openrouter", primary_model=None,
                 fallback_model=None, api_keys=None, confirm=console_confirm,
                 confirm_tools=("run_periodic",), max_steps=25, max_tokens=4000,
                 retries=2, log=print, backends=None):
        """
        runner        : runner.Runner
        primary / fallback : llm.make_backend providers ('gemini' | 'openrouter');
                        fallback=None disables switching
        api_keys      : optional {'gemini': key, 'openrouter': key} (else env vars)
        confirm       : fn(tool, args, plan) -> bool for tools in confirm_tools;
                        None = no approval step
        backends      : pre-built backends (tests / custom servers), overrides the above
        """
        self.runner, self.log = runner, log
        self.confirm, self.confirm_tools = confirm, set(confirm_tools or ())
        self.max_steps, self.max_tokens, self.retries = max_steps, max_tokens, retries
        keys = api_keys or {}
        self._specs = [] if backends else [(p, m, keys.get(p)) for p, m in
                                          ((primary, primary_model), (fallback, fallback_model)) if p]
        self._backends = list(backends) if backends else [None] * len(self._specs)
        self.messages: list[dict] = []
        self._active = 0                                 # backend in use for the current turn
        self.tools = {
            "get_settings": self._get_settings, "update_settings": self._update_settings,
            "list_areas": self._list_areas, "plan_periodic_run": self._plan,
            "run_periodic": self._run_periodic, "run_recent": self._run_recent,
            "quick_check": self._quick_check, "list_outputs": self._list_outputs,
            "read_summary": self._read_summary, "last_run": self._last_run,
        }

    # ------------------------------------------------------------ LLM plumbing
    def _backend(self, i):
        if self._backends[i] is None:
            p, m, k = self._specs[i]
            self._backends[i] = make_backend(p, m, k)
        return self._backends[i]

    def _system(self):
        return SYSTEM_PROMPT.format(
            today=pd.Timestamp.today().date(),
            stages="; ".join(f"{k} {v}" for k, v in ph.STAGE_CLASSES.items()),
            qc="; ".join(f"{k} {v}" for k, v in ph.QC_DESCRIPTION.items()))

    def _complete(self):
        last_err = None
        for i in range(self._active, len(self._backends)):
            try:
                be = self._backend(i)
            except Exception as e:                       # e.g. fallback key missing
                last_err = e
                self.log(f"[agent] backend {i} unavailable: {e}")
                continue
            for attempt in range(self.retries + 1):
                try:
                    out = be.complete(self._system(), self.messages, TOOLS, self.max_tokens)
                    self._active = i                     # stay on it for the rest of this turn
                    return out
                except TransientLLMError as e:
                    last_err = e
                    if attempt < self.retries:
                        time.sleep(2 * 2 ** attempt)
            if i + 1 < len(self._backends):
                self.log(f"[agent] {be.name} unavailable ({last_err}); switching to the fallback model")
        raise RuntimeError(f"no LLM backend available: {last_err}")

    def reset(self):
        self.messages = []

    def chat(self, text: str) -> str:
        """One user turn: the agent calls tools until it has an answer."""
        self.messages.append({"role": "user", "content": text})
        self._active = 0                                 # each new request tries the primary first
        for _ in range(self.max_steps):
            out = self._complete()
            if out["message"] is not None:
                self.messages.append(out["message"])
            if not out["tool_calls"]:
                return out["text"]
            for call in out["tool_calls"]:
                result = self._call_tool(call["name"], call.get("args") or {})
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                      "content": _clip_json(result), "_gemini_id": call.get("gemini_id")})
        return "(stopped: too many tool steps for one request — ask me to continue)"

    def _call_tool(self, name, args):
        fn = self.tools.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}"}
        args = _coerce(_SCHEMAS.get(name), args)
        self.log(f"[agent] {name}({json.dumps(args, default=str)[:300]})")
        try:
            if name in self.confirm_tools and self.confirm is not None:
                plan = self._plan(**{k: args[k] for k in ("level", "names", "as_of") if k in args}) \
                    if name == "run_periodic" else None
                if not self.confirm(name, args, plan):
                    return {"status": "cancelled", "reason": "the user did not approve this run"}
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"{e.__class__.__name__}: {e}"}

    # ------------------------------------------------------------ tools
    def _get_settings(self):
        c = self.runner.cfg.public()
        c["output_destination"] = self.runner.store.describe()
        return c

    def _update_settings(self, pheno=None, **changes):
        # only the fields the tool declares — paths, bucket and Drive folder stay
        # as the operator configured them
        allowed = set(_SCHEMAS["update_settings"]["properties"]) - {"pheno"}
        extra = sorted(set(changes) - allowed)
        if extra:
            return {"error": f"cannot change {extra} from the agent; allowed: {sorted(allowed)}"}
        if pheno:
            changes["pheno_cfg"] = pheno
        self.runner.update(**changes)
        return {"status": "updated", "settings": self._get_settings()}

    def _list_areas(self, kind, region=None, contains=None):
        names = self.runner.list_areas(kind, region, contains)
        return {"kind": kind, "count": len(names), "names": names[:300]}

    def _plan(self, level, names=None, as_of=None):
        plan = self.runner.plan_periodic(level, names, as_of)
        plan["detail"] = plan["detail"][:100]
        return plan

    def _run_periodic(self, level, names=None, as_of=None, skip_existing=True, only=None):
        res = self.runner.run_periodic(level, names, as_of, skip_existing=skip_existing, only=only)
        out = {"as_of": res["as_of"], "output": self.runner.store.describe(),
               "counts": {k: len(v) for k, v in res.items() if isinstance(v, list)}}
        for k in ("success", "failed", "no_coverage", "out_of_season", "skipped"):
            out[k] = res[k][:60]                         # keep the reply within the context budget
        return out

    def _run_recent(self, names=None, bbox=None, as_of=None, lookback_days=None):
        if not names and not bbox:
            return {"error": "give names (municipal / barangay) or a bbox"}
        return self.runner.run_recent(names, bbox, as_of, lookback_days)

    def _quick_check(self, bbox, planting_month=None, year=None):
        return self.runner.quick_check(bbox, planting_month, year)

    def _list_outputs(self, kind, contains=None):
        return self.runner.list_outputs(kind, contains)

    def _read_summary(self, path):
        return {"path": path, "rows": self.runner.read_summary(path)}

    def _last_run(self, kind):
        res = self.runner.last_periodic if kind == "periodic" else self.runner.last_recent
        return res or {"status": f"no {kind} run yet in this session"}
