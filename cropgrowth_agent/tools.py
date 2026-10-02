"""Tools the agent can call. Each returns a JSON-serialisable dict.

TOOLS holds the schemas (Gemini function-declaration format, also sent to
OpenAI-compatible models); ToolRunner implements them as t_<name> methods on
top of runner.Runner, so every number the agent reports comes from the pipeline.
"""
from __future__ import annotations

import json

from . import pipeline as pl

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
         "season": {"type": "string", "enum": list(pl.SEASONS),
                    "description": "dry (Semester_1 planting months, ~Oct-Dec) or wet "
                                   "(Semester_2, ~May-Jun); switches the planting-month column"},
         "season_year": {"type": "integer",
                         "description": "harvest year naming the season: dry2026 = planted "
                                        "Oct-Dec 2025; wet2026 = planted May-Jun 2026"},
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
    {"name": "search_method_docs",
     "description": "Search this project's documentation and code (README, module and function "
                    "docstrings) for how the method works: thresholds, stage rules, QC codes, data "
                    "sources, outputs. Use for questions about the method, not to run it.",
     "parameters": {"type": "object", "properties": {
         "query": _STR, "k": {"type": "integer"}}, "required": ["query"]}},
    {"name": "last_run",
     "description": "Result of the last periodic or recent run in this session (successes, "
                    "failures with errors, outputs).",
     "parameters": {"type": "object", "properties": {
         "kind": {"type": "string", "enum": ["periodic", "recent"]}}, "required": ["kind"]}},
]

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


def public(obj):
    """Drop private keys (e.g. '_data' xarray payloads) before results go to the model."""
    if isinstance(obj, dict):
        return {k: public(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [public(v) for v in obj]
    return obj


# ---------------------------------------------------------------- implementations
class ToolRunner:
    """runner  : runner.Runner (settings, data, output store)
    confirm : fn(tool, args, plan) -> bool, asked before the tools in confirm_tools;
              None = no approval step
    docs    : docs.DocsIndex for search_method_docs (None = unavailable)"""

    def __init__(self, runner, confirm=console_confirm, confirm_tools=("run_periodic",),
                 docs=None, log=print):
        self.runner, self.confirm, self.docs, self.log = runner, confirm, docs, log
        self.confirm_tools = set(confirm_tools or ())

    def __call__(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"t_{name}", None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        args = _coerce(_SCHEMAS.get(name), args or {})
        try:
            if name in self.confirm_tools and self.confirm is not None:
                plan = self.t_plan_periodic_run(**{k: args[k] for k in ("level", "names", "as_of")
                                                   if k in args}) if name == "run_periodic" else None
                if not self.confirm(name, args, plan):
                    return {"status": "cancelled", "reason": "the user did not approve this run"}
            return public(fn(**args))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    # -----------------------------------------------------------
    def t_get_settings(self):
        c = self.runner.cfg.public()
        c["output_destination"] = self.runner.store.describe()
        c["season_resolved"] = self.runner.cfg.season_summary()
        return c

    def t_update_settings(self, pheno=None, **changes):
        # only the fields the tool declares — paths, bucket and Drive folder stay
        # as the operator configured them
        allowed = set(_SCHEMAS["update_settings"]["properties"]) - {"pheno"}
        extra = sorted(set(changes) - allowed)
        if extra:
            return {"error": f"cannot change {extra} from the agent; allowed: {sorted(allowed)}"}
        if pheno:
            changes["pheno_cfg"] = pheno
        self.runner.update(**changes)
        return {"status": "updated", "settings": self.t_get_settings()}

    def t_list_areas(self, kind, region=None, contains=None):
        names = self.runner.list_areas(kind, region, contains)
        return {"kind": kind, "count": len(names), "names": names[:300]}

    def t_plan_periodic_run(self, level, names=None, as_of=None):
        plan = self.runner.plan_periodic(level, names, as_of)
        plan["detail"] = plan["detail"][:100]
        return plan

    def t_run_periodic(self, level, names=None, as_of=None, skip_existing=True, only=None):
        res = self.runner.run_periodic(level, names, as_of, skip_existing=skip_existing, only=only)
        out = {"as_of": res["as_of"], "output": self.runner.store.describe(),
               "counts": {k: len(v) for k, v in res.items() if isinstance(v, list)}}
        for k in ("success", "failed", "no_coverage", "out_of_season", "skipped"):
            out[k] = res[k][:60]                         # keep the reply within the context budget
        return out

    def t_run_recent(self, names=None, bbox=None, as_of=None, lookback_days=None):
        if not names and not bbox:
            return {"error": "give names (municipal / barangay) or a bbox"}
        return self.runner.run_recent(names, bbox, as_of, lookback_days)

    def t_quick_check(self, bbox, planting_month=None, year=None):
        return self.runner.quick_check(bbox, planting_month, year)

    def t_list_outputs(self, kind, contains=None):
        return self.runner.list_outputs(kind, contains)

    def t_read_summary(self, path):
        return {"path": path, "rows": self.runner.read_summary(path)}

    def t_search_method_docs(self, query, k=5):
        if self.docs is None:
            return {"error": "method docs index not loaded"}
        return {"results": self.docs.search(query, k=k)}

    def t_last_run(self, kind):
        res = self.runner.last_periodic if kind == "periodic" else self.runner.last_recent
        return res or {"status": f"no {kind} run yet in this session"}
