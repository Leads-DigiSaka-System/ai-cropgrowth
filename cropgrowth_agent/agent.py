"""Natural-language agent: "map the current growth stage of Muñoz" → saved stage maps + hectares.

Gemini (or Qwen via OpenRouter when Gemini is busy, see llm.py) plans and calls the pipeline tools
(tools.py → runner.Runner); this repository's own docs are searched when it needs to explain the method."""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from . import phenology as ph
from .llm import GEMINI_FALLBACKS, ModelUnavailableError, TransientLLMError, make_backend
from .tools import TOOLS, ToolRunner, console_confirm

SYSTEM = """You operate the rice growth-stage mapping pipeline for the Philippines: satellite NDVI
(Harmonized Landsat Sentinel-2 or Sentinel-2) → cloud mask → 10-day composites → cropland mask →
Savitzky-Golay smoothing → phenology dates (NDVI 15% threshold + derivatives) → growth-stage maps.
Today is {today}.

Products
- Periodic mode (national / regional / provincial areas): one stage map per province per period
  (monthly, or semimonthly = 1st-15th and 16th-end), season-based. Tools: plan_periodic_run, run_periodic.
  Seasons: dry (planting month column Semester_1, planted ~Oct-Dec, harvested the next year) and
  wet (Semester_2, planted ~May-Jun). A season is named by its harvest year (dry2026 = planted
  Oct-Dec 2025). Each province's window runs from 1 month before its planting month to 6 after.
  A period outside a province's season window is reported as out_of_season — if the user asks for
  months of the other season, switch with update_settings(season=..., season_year=...).
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
   shares per stage, data age, and every failure with its error. Never invent numbers or paths.
6. For failed provinces caused by network / catalog errors, offer to retry them with
   run_periodic(only=[...]), then rebuild the national / regional mosaic with mosaic_periodic.
   If a tool returns an error, read it and fix the call or explain.
7. For questions about how the method works, use search_method_docs and cite the source it came from.
Be concise. Use plain language; the users are agriculture staff."""


class CropGrowthAgent:
    """runner: runner.Runner (built from `config` + `gcs_client` when not given).
    provider: "gemini" (default) or "openrouter" (Qwen via OpenRouter, key in OPENROUTER_API_KEY).
    When the active model stays busy (rate limit / overload / network) after `retries`, or doesn't exist for
    this key (404), the same conversation continues on the next model in the chain:
      primary → gemini_fallbacks (default gemini-3.5-flash-lite) → Qwen on OpenRouter.
    fallback: "auto" = add Qwen if OPENROUTER_API_KEY is set, "openrouter" = always, None = no Qwen.
    Each new request tries the primary model first again.
    confirm: fn(tool, args, plan) -> bool asked before the tools in confirm_tools (default: a y/N prompt
    before run_periodic); None = no approval step (e.g. scheduled runs)."""

    def __init__(self, runner=None, config=None, gcs_client=None, llm_model: str | None = None,
                 api_key: str | None = None, provider: str = "gemini", fallback: str | None = "auto",
                 fallback_model: str | None = None, retries: int = 2, gemini_fallbacks: list[str] | None = None,
                 backend=None, fallback_backends: list | None = None, confirm=console_confirm,
                 confirm_tools=("run_periodic",), docs=None, verbose: bool = True, max_steps: int = 25):
        self.verbose, self.max_steps, self.retries = verbose, max_steps, retries
        if runner is None:
            from .runner import RunConfig, Runner
            runner = Runner(config or RunConfig(), gcs_client=gcs_client, log=self._log)
        self.runner = runner
        self.backend = backend or make_backend(provider, llm_model, api_key)
        if fallback_backends is None:
            fallback_backends = []
            if provider == "gemini":
                for m in GEMINI_FALLBACKS if gemini_fallbacks is None else gemini_fallbacks:
                    if m != self.backend.model:
                        fallback_backends.append(make_backend("gemini", m, api_key))
                if fallback and (fallback == "openrouter" or os.environ.get("OPENROUTER_API_KEY")):
                    fallback_backends.append(make_backend("openrouter", fallback_model))
        self.fallbacks = list(fallback_backends)
        self._sleep = time.sleep
        if docs is None:
            docs = self._load_docs()
        self.tools = ToolRunner(runner, confirm=confirm, confirm_tools=confirm_tools,
                                docs=docs or None, log=self._log)
        self.messages: list[dict] = []   # provider-neutral history (see llm.py)
        self._active = 0
        self._log(f"LLM: {self.backend.name}"
                  + (f" (fallback: {' → '.join(b.name for b in self.fallbacks)})" if self.fallbacks else ""))

    @staticmethod
    def _load_docs():
        try:
            from .docs import DocsIndex
            return DocsIndex.build()
        except Exception as e:
            print(f"(method docs search unavailable: {e})")
            return None

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _system(self) -> str:
        return SYSTEM.format(
            today=pd.Timestamp.today().date(),
            stages="; ".join(f"{k} {v}" for k, v in ph.STAGE_CLASSES.items()),
            qc="; ".join(f"{k} {v}" for k, v in ph.QC_DESCRIPTION.items())) + (
            f"\nOutputs are saved to: {self.runner.store.describe()}")

    # -------------------------------------------------------------- main loop
    def _complete(self, system: str) -> dict:
        """Ask the active model; on a transient error retry with backoff, then move to the fallback."""
        chain = [self.backend] + self.fallbacks
        err = None
        for i in range(self._active, len(chain)):
            b = chain[i]
            for attempt in range(self.retries + 1):
                try:
                    return b.complete(system, self.messages, TOOLS)
                except ModelUnavailableError as e:
                    err = e
                    self._log(f"  {b.name} is not available for this API key ({str(e)[:160]})")
                    break
                except TransientLLMError as e:
                    err = e
                    if attempt < self.retries:
                        wait = 2 ** (attempt + 1)
                        self._log(f"  {b.name} unavailable ({str(e)[:120]}); retrying in {wait}s")
                        self._sleep(wait)
            if i + 1 < len(chain):
                self._log(f"  {b.name} unavailable → switching to {chain[i + 1].name}")
                self._active = i + 1
        raise err

    def run(self, request: str) -> str:
        """Send a request; the agent calls tools until it has an answer. Keeps the
        conversation, so follow-ups ("save it to Drive instead", "retry the failed ones") work."""
        system = self._system()
        self.messages.append({"role": "user", "content": request})
        self._active = 0                                   # every request tries the primary model first
        for _ in range(self.max_steps):
            try:
                res = self._complete(system)
            except TransientLLMError as e:
                return f"No model available right now ({e}). Try again in a few minutes."
            if res["message"] is None:
                return res["text"]
            self.messages.append(res["message"])
            if not res["tool_calls"]:
                return res["text"]
            for c in res["tool_calls"]:
                self._log(f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)[:200]})")
                out = self.tools(c["name"], c["args"])
                if "error" in out:
                    self._log(f"  ! {out['error']}")
                dumped = json.dumps(out, default=str)          # plain JSON types only (no Path/numpy)
                if len(dumped) > 20000:
                    dumped = json.dumps({"truncated": True, "preview": dumped[:20000]})
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                      "content": dumped, "_gemini_id": c.get("gemini_id")})
        return "Stopped: too many steps without finishing."

    def reset(self):
        self.messages = []

    def chat(self):
        print("What should I map? (empty line to quit)")
        while True:
            q = input("\n> ").strip()
            if not q:
                break
            print("\n" + self.run(q))
