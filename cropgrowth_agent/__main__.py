"""Command line.

  # talk to it (needs GEMINI_API_KEY; with OPENROUTER_API_KEY set it falls back to Qwen when Gemini is busy)
  python -m cropgrowth_agent --config config.json "map the current growth stage of Science City of Muñoz"
  python -m cropgrowth_agent --config config.json --chat
  python -m cropgrowth_agent --provider openrouter --config config.json --chat       # Qwen only

  # or run the pipeline directly, no LLM involved
  python -m cropgrowth_agent areas --config config.json --kind region
  python -m cropgrowth_agent plan --config config.json --level regional --names "REGION III"
  python -m cropgrowth_agent periodic --config config.json --level provincial --names BOHOL --as-of 2026-03-01
  python -m cropgrowth_agent recent --config config.json --names "SCIENCE CITY OF MUÑOZ"
  python -m cropgrowth_agent recent --config config.json --bbox 120.90 15.60 120.95 15.65
  python -m cropgrowth_agent quick-check --config config.json --bbox 120.90 15.60 120.93 15.63 --planting-month 12
  python -m cropgrowth_agent outputs --config config.json --kind periodic

Settings come from --config (a JSON file of RunConfig fields, see runner.py) and the flags below,
which override it. Output destination: --output-target gcs | gdrive | both | local.
"""
import argparse
import json
import os
import sys

DIRECT = ("areas", "plan", "periodic", "recent", "quick-check", "outputs")

# flag -> RunConfig field
SETTINGS = {
    "--vector-path": ("vector_path", str), "--prov-col": ("prov_col", str),
    "--region-col": ("region_col", str), "--plant-mo-col": ("plant_mo_col", str),
    "--aoi-path": ("aoi_path", str), "--aoi-name-col": ("aoi_name_col", str),
    "--data-source": ("data_source", str), "--resolution": ("resolution_m", float),
    "--tile-deg": ("tile_deg", float), "--tile-workers": ("tile_workers", int),
    "--tile-cache-dir": ("tile_cache_dir", str), "--year": ("year", int),
    "--cadence": ("cadence", str), "--periods": ("periods", str),
    "--cog-label": ("cog_label", str), "--output-prefix": ("output_prefix", str),
    "--recent-prefix": ("recent_prefix", str), "--lookback-days": ("lookback_days", int),
    "--output-target": ("output_target", str), "--gcs-bucket": ("gcs_bucket", str),
    "--drive-root": ("drive_root", str), "--local-root": ("local_root", str),
}


def _add_settings(p):
    p.add_argument("--config", default=os.environ.get("CROPGROWTH_CONFIG"),
                   help="JSON file of RunConfig fields (or CROPGROWTH_CONFIG)")
    for flag, (field, typ) in SETTINGS.items():
        p.add_argument(flag, dest=field, type=typ)
    p.add_argument("--gcs-project", help="Google Cloud project for the GCS client")


def _runner(a, log=print):
    from .runner import RunConfig, Runner
    cfg = {}
    if a.config:
        with open(a.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    for field, _ in SETTINGS.values():
        v = getattr(a, field, None)
        if v is not None:
            cfg[field] = v
    if "exclude" in cfg:
        cfg["exclude"] = tuple(cfg["exclude"])
    rc = RunConfig(**cfg)
    client = None
    if rc.output_target in ("gcs", "both"):
        from google.cloud import storage                     # application-default credentials
        client = storage.Client(project=a.gcs_project)
    return Runner(rc, gcs_client=client, log=log)


def _print(obj):
    from .tools import public
    print(json.dumps(public(obj), indent=2, default=str, ensure_ascii=False))


def direct(argv):
    p = argparse.ArgumentParser(prog="cropgrowth_agent")
    p.add_argument("cmd", choices=DIRECT)
    _add_settings(p)
    p.add_argument("--kind", default=None, help="areas: province | region | aoi; outputs: periodic | recent")
    p.add_argument("--level", choices=["national", "regional", "provincial"], default="national")
    p.add_argument("--names", nargs="+", help="regions / provinces (periodic, plan) or AOI names (recent)")
    p.add_argument("--as-of", help="YYYY-MM-DD (default today)")
    p.add_argument("--only", nargs="+", help="periodic: only these provinces (e.g. retry failures)")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    p.add_argument("--planting-month", type=int)
    p.add_argument("--contains")
    a = p.parse_args(argv)
    r = _runner(a)
    if a.cmd == "areas":
        _print(r.list_areas(a.kind or "province", contains=a.contains))
    elif a.cmd == "plan":
        _print(r.plan_periodic(a.level, a.names, a.as_of))
    elif a.cmd == "periodic":
        _print(r.run_periodic(a.level, a.names, a.as_of, skip_existing=not a.no_skip_existing, only=a.only))
    elif a.cmd == "recent":
        if not a.names and not a.bbox:
            p.error("recent needs --names or --bbox")
        _print(r.run_recent(a.names, a.bbox, a.as_of))
    elif a.cmd == "quick-check":
        if not a.bbox:
            p.error("quick-check needs --bbox")
        _print(r.quick_check(a.bbox, a.planting_month))
    else:
        _print(r.list_outputs(a.kind or "periodic", a.contains))


def main():
    if len(sys.argv) > 1 and sys.argv[1] in DIRECT:
        return direct(sys.argv[1:])

    p = argparse.ArgumentParser(prog="cropgrowth_agent")
    p.add_argument("request", nargs="?")
    p.add_argument("--chat", action="store_true")
    _add_settings(p)
    p.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini",
                   help="gemini (GEMINI_API_KEY) or openrouter (Qwen, OPENROUTER_API_KEY)")
    p.add_argument("--model", help="model id (default gemini-3.5-flash, or the Qwen model for openrouter)")
    p.add_argument("--gemini-fallback", action="append",
                   help="Gemini model(s) to try before Qwen (default gemini-3.5-flash-lite; 'none' to skip)")
    p.add_argument("--fallback", choices=["auto", "openrouter", "none"], default="auto",
                   help="when Gemini stays busy, continue on Qwen via OpenRouter "
                        "(auto: if OPENROUTER_API_KEY is set)")
    p.add_argument("--fallback-model", help="OpenRouter model for the fallback (default qwen/qwen3-235b-a22b-2507)")
    p.add_argument("--yes", action="store_true", help="run national / regional / provincial jobs without asking")
    a = p.parse_args()
    from .agent import CropGrowthAgent
    from .tools import console_confirm
    agent = CropGrowthAgent(runner=_runner(a), llm_model=a.model, provider=a.provider,
                            fallback=None if a.fallback == "none" else a.fallback,
                            fallback_model=a.fallback_model,
                            gemini_fallbacks=None if not a.gemini_fallback else
                            [m for m in a.gemini_fallback if m != "none"],
                            confirm=None if a.yes else console_confirm)
    if a.chat or not a.request:
        agent.chat()
    else:
        print(agent.run(a.request))


if __name__ == "__main__":
    main()
