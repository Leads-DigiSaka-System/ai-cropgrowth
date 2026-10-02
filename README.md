# Rice growth-stage agent

Ask it in plain language, for example:

> "What is the current growth stage of rice in Science City of Muñoz? Save the map to Google Drive."

> "Make last month's maps for the provinces of Region III."

and it produces, in Google Drive, Cloud Storage or both:

```
products/growth_stage/recent/                      municipal / barangay: the stage TODAY
├── SCIENCE_CITY_OF_MUÑOZ_20261002.tiff            growth_stage, data_age_days, qc (3 bands, 30 m)
├── SCIENCE_CITY_OF_MUÑOZ_20261002_summary.csv     pixels, hectares, % of cropland per stage
└── runs/recent_<timestamp>.csv                    what ran, where it went, failures

products/growth_stage/2026/dry/                    provincial / regional / national: one map per period
├── dry2026_BULACAN_202609.tiff                    monthly (…_202609H1 / H2 when semimonthly)
├── summary/dry2026_BULACAN_202609.csv             hectares per stage
├── qa/dry2026_BULACAN_phenology_dates.tiff        optional: the 8 transition dates + QC bands
└── runs/periodic_<timestamp>.csv
```

Stage classes in every map:

| Value | Stage |
|---|---|
| −1 | not cropland / outside the area |
| 0 | no rice cycle detected |
| 1 | pre-planting / transplanting |
| 2 | vegetative (emergence → panicle initiation) |
| 3 | reproductive (panicle initiation → peak NDVI) |
| 4 | ripening / maturity (peak → harvest) |
| 5 | harvested / post-harvest |

## How a request runs

1. **Pick the product from the area.**
   - Municipality, barangay or a small bbox → **recent** (`run_recent`): the current stage.
   - Province, region or the whole country → **periodic** (`plan_periodic_run`, `run_periodic`): one map per
     province per month or half-month.
2. **`list_areas`** checks the place name against your boundary files. If it is ambiguous or missing, the agent asks.
3. **`plan_periodic_run`** (periodic only) is a dry run: provinces, tiles, periods and destination, with nothing
   downloaded. `run_periodic` then **asks you to approve** before it starts.
4. **`update_settings`** switches the destination when you say "save to Drive" or "to Cloud Storage".
5. The pipeline runs (below). The agent reports hectares per stage, data age, output links and every failure.
   It can retry failed provinces with `run_periodic(only=[...])`.

Other tools:
- **`quick_check`** runs on a small bbox and returns the QC counts, to check thresholds.
- **`list_outputs`** / **`read_summary`** show what's saved and the hectares per stage.
- **`last_run`** returns the last run's results.
- **`search_method_docs`** answers "how does it detect X" from this repository's docs and code, with the source.

Every number the agent reports comes from a tool result. It can only change the settings the `update_settings`
tool declares, so file paths, the bucket and the Drive folder stay as you configured them.

## The pipeline

```
HLS v2 (Landsat 8/9 + Sentinel-2) or Sentinel-2 L2A  (Microsoft Planetary Computer)
→ Fmask / SCL cloud mask → 10-day NDVI composites → ESA WorldCover cropland mask
→ Savitzky-Golay smoothing → NDVI 15 % amplitude threshold + derivatives
→ transition dates (planting, emergence, tillering, panicle initiation, heading, peak, maturity, harvest)
→ growth stage per date or period
```

| | **Periodic** | **Recent** |
|---|---|---|
| Areas | national, regional, provincial | municipal, barangay, bbox |
| When | on a schedule: monthly or semimonthly (1st–15th, 16th–end) | on demand |
| Window | each province's season (planting month −1 … +6), data up to `as_of` | rolling 240 days ending today |
| Crop cycle | the season's cycle (highest NDVI peak) | the **most recent** cycle (`PEAK_SELECT='last'`) |
| Default period | the newest one that has ended (a monthly run on 1 March makes February) | — |

- **Periodic, other periods:** `periods='current'` maps the period still in progress (provisional); `'season'`
  backfills every finished period of the season. Provinces whose season doesn't include the period are reported
  as `out_of_season`.
- **Recent:** composites are anchored on today, so the newest scenes count. Following the latest cycle means a
  young crop isn't mistaken for the previous harvested one. `data_age_days` shows how stale each pixel's stage
  is: Sentinel-2 / Landsat revisit every few days, but clouds can hide a field for weeks.

### Seasons

The planting-month column of the province file chooses the season:

| `plant_mo_col` | Season | Usual planting | Named by its harvest year (`season_year`) | Fallback month |
|---|---|---|---|---|
| `Semester_1` | dry | Oct–Dec (some provinces Jan–Mar) | `dry2026` = planted Oct–Dec 2025 or Jan–Mar 2026 | 12 |
| `Semester_2` | wet | May–Jun | `wet2026` = planted May–Jun 2026 | 6 |

- **Per-province window:** each province is processed from one month before its planting month to six months
  after. For dry2026, a November planting gives 2025-10-01 to 2026-05-31, and a January planting gives
  2025-12-01 to 2026-07-31. For wet2026, a June planting gives 2026-05-01 to 2026-12-31.
- **Planting-month values:** numbers (`5`, `5.0`, `"5"`), month names (`May`, `June`) and ranges (`5-6`,
  `May-June`, which use the first month) are all read. An empty or unreadable cell uses the season's fallback
  month, and the plan flags it.
- **File names and folder:** follow the season (`wet2026_<PROVINCE>_202609.tiff` under
  `products/growth_stage/2026/wet`) unless you set `cog_label` / `output_prefix`.
- **Out of season:** a period outside a province's window is reported as `out_of_season`. For example, a monthly
  run in October makes September maps for wet2026 but none for dry2026.
- **Switching seasons:** to switch, change `plant_mo_col` (or set `season='wet'`) and `season_year`. The agent
  does the same when asked ("switch to the wet season").

### Data sources

| | **`hls`** (default): Harmonized Landsat Sentinel-2 v2.0 | **`s2`**: Sentinel-2 L2A |
|---|---|---|
| Sensors | Landsat 8/9 (`hls2-l30`) + Sentinel-2 (`hls2-s30`) | Sentinel-2 |
| Revisit | ~2–3 days combined | 5 days |
| Native resolution | 30 m | 10 m (red/NIR) |
| Cloud mask | Fmask: drops cirrus, cloud, adjacent cloud, shadow, snow, high aerosol; keeps water (flooded paddies) | SCL classes 4/5/6 |
| NIR band | B8A (S30) / B05 (L30) | B08 |

- **HLS** gives more clear looks through cloud, which is the main limit for rice phenology in the Philippines.
- **S2** is the choice for a 10 m grid, and may be a few days fresher for recent maps.
- **Re-check thresholds when switching:** HLS NDVI uses the narrow NIR band, so confirm `MIN_PEAK_NDVI` and
  `MAX_BASE_NDVI` with `quick_check`.
- **Grid:** 30 m by default (`resolution_m`). At 30 m, Sentinel-2 red/NIR are averaged from the 10 m bands
  (reading the COG overviews), and the cloud and cropland classes take the majority.

### Speed and reliability

- **Searches and signing:** one STAC search per collection per province, filtered per tile locally. Asset URLs are
  signed at read time, so the search stays valid for a multi-hour run.
- **Cropland first:** each tile reads the WorldCover map before any satellite data. Tiles with no cropland are
  skipped, and imagery is read only over the cropland's extent. Tiles outside the province outline (sea) are
  never processed.
- **Parallel tiles:** `tile_workers`, ~1–1.5 GB RAM each.
- **Unreadable scenes:** some Planetary Computer files open without a coordinate system (`AssertionError: src.crs
  is not None`, every retry). The band headers are then probed, and those scenes are dropped, logged
  (`dropping N unreadable scene(s): <id>`) and skipped for the rest of the run.
- **Failures and resume:** each tile gets 2 extra attempts. A tile that still fails marks the province as failed,
  so the next run retries it. With `tile_cache_dir`, finished tiles are reused on that rerun. The cache key covers
  the window, grid, compositing, cloud settings and phenology parameters, so a new period or setting never
  reuses stale tiles.

## Output destinations

`output_target` (or tell the agent "save to Drive"):

| Target | Where | Needs |
|---|---|---|
| `gdrive` | `drive_root/<path>` (default `MyDrive/AI-CropGrowth/outputs`) | Drive mounted in Colab, or Google Drive for desktop |
| `gcs` | `gs://gcs_bucket/<path>` | a `google.cloud.storage.Client` |
| `both` | both of the above; a file counts as existing only when both have it | both |
| `local` | `local_root/<path>` | nothing |

## Run it

**Colab:** open `notebooks/CropGrowth_Agent_Colab.ipynb` (the agent).
`notebooks/run_growth_stages.ipynb` is the full notebook: settings, quick-check plots, periodic / recent runs without
the LLM, and the agent.

**Command line:**

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
export OPENROUTER_API_KEY=...             # optional: Qwen fallback when Gemini is busy
python -m cropgrowth_agent --config config.json "what is the current stage of rice in Science City of Muñoz"
python -m cropgrowth_agent --config config.json --chat

# same pipeline without the LLM
python -m cropgrowth_agent areas       --config config.json --kind region
python -m cropgrowth_agent plan        --config config.json --level regional --names "REGION III"
python -m cropgrowth_agent periodic    --config config.json --level provincial --names BULACAN --as-of 2026-10-01
python -m cropgrowth_agent recent      --config config.json --names "SCIENCE CITY OF MUÑOZ"
python -m cropgrowth_agent quick-check --config config.json --bbox 120.90 15.60 120.93 15.63 --planting-month 12
python -m cropgrowth_agent outputs     --config config.json --kind periodic
```

`config.json` holds `RunConfig` fields (`cropgrowth_agent/runner.py`); any flag overrides it:

```json
{
  "vector_path": "/data/vector/boundary_province_philippines.gpkg",
  "prov_col": "Pro_Name", "region_col": "Reg_Name", "plant_mo_col": "Semester_2",
  "aoi_path": "/data/vector/boundary_municipal_philippines.gpkg", "aoi_name_col": "Mun_Name",
  "season_year": 2026,
  "cadence": "monthly", "data_source": "hls", "resolution_m": 30,
  "output_target": "gdrive", "drive_root": "/content/drive/MyDrive/AI-CropGrowth/outputs",
  "tile_cache_dir": "/data/tile_cache"
}
```

**Main settings** (`RunConfig`):

| Setting | Default | What it does |
|---|---|---|
| `data_source` | `hls` | `hls` or `s2` |
| `resolution_m` | 30 | Output pixel size; `tile_deg` follows it (0.25° at 30 m, 0.1° at 10 m) |
| `cadence` | `monthly` | `monthly` or `semimonthly` |
| `periods` | `last_complete` | `last_complete`, `current` or `season` |
| `stage_method` | `dominant` | Stage that fills most of the period, or `midpoint` |
| `plant_mo_col` | `Semester_1` | Planting-month column: `Semester_1` = dry season, `Semester_2` = wet season |
| `season_year` | season in progress | Harvest year naming the season (`dry2026`, `wet2026`) |
| `planting_month_fallback` | 12 dry / 6 wet | Month used when a province has no planting month |
| `lookback_days` | 240 | Recent-mode window |
| `output_target` | `gcs` | `gcs`, `gdrive`, `both` or `local` |
| `pheno_cfg` | see `runner.DEFAULT_PHENO_CFG` | Phenology thresholds (all in `phenology.DEFAULTS`) |
| `tile_workers` | 2 | Tiles in parallel |
| `tile_cache_dir` | none | Resume cache for interrupted provinces |

Python, without the CLI:

```python
from cropgrowth_agent import RunConfig, Runner, CropGrowthAgent
runner = Runner(RunConfig(vector_path=..., aoi_path=..., output_target="gdrive"))
runner.run_recent(["SCIENCE CITY OF MUÑOZ"])                   # direct
agent = CropGrowthAgent(runner=runner)
print(agent.run("Make last month's maps for Region III"))      # through the agent
```

## Repository layout

```
cropgrowth_agent/   data_processing.py (imagery → NDVI cubes, tiling, cropland mask), phenology.py (dates, stages,
                    COG export), pipeline.py (periods, admin units, periodic / recent runs), runner.py (end-to-end
                    runs + saving), output_store.py (Drive / GCS / local), tools.py + agent.py (the agent),
                    llm.py (Gemini / Qwen), docs.py (method docs search), CLI
notebooks/          CropGrowth_Agent_Colab.ipynb (the agent), run_growth_stages.ipynb (full pipeline notebook)
tests/              pytest; runs offline with a fake Planetary Computer (no network, odc-stac or API keys)
```

Run the tests with `pip install pytest && pytest`.

## Language model: Gemini with a Qwen fallback

The agent uses Gemini 3.5 Flash by default. When a model keeps returning rate-limit, overload or network errors
(after 2 retries, 2 s then 4 s apart), or isn't available for your key (404, e.g. a retired model), the same
conversation continues on the next model, including the tool results so far:

`gemini-3.5-flash` → `gemini-3.5-flash-lite` → Qwen via OpenRouter (only when `OPENROUTER_API_KEY` is set)

The next request tries the first model again.

| Option | Default | What it does |
|---|---|---|
| `--provider` | `gemini` | `openrouter` uses Qwen only (needs just `OPENROUTER_API_KEY`) |
| `--model` | `gemini-3.5-flash` | Primary model id (`GEMINI_MODEL` also works) |
| `--gemini-fallback` | `gemini-3.5-flash-lite` | Gemini model(s) tried before Qwen; repeatable, `none` to skip |
| `--fallback` | `auto` | `auto`: Qwen when `OPENROUTER_API_KEY` is set; `openrouter`: always; `none`: off |
| `--fallback-model` | `qwen/qwen3-235b-a22b-2507` | Any OpenRouter model with tool calling (`OPENROUTER_MODEL` also works) |
| `--yes` | off | Run national / regional / provincial jobs without the approval prompt (scheduled runs) |

Keys are read from the environment (or Colab secrets in the notebooks), never from files in the repo.

## Method reference

The rule for each transition date (NDVI₁₅ crossings, maximum growth rate, maximum acceleration, steepest decline,
…) is in the `cropgrowth_agent/phenology.py` module docstring. QC codes: `0` complete cycle, `1` ongoing past peak,
`2` ongoing before peak; `10` too few clear observations, `11` no interior peak, `12` amplitude too small, `13` peak
NDVI too low, `14` base NDVI too high, `15` no rising 15 % crossing, `16` season length implausible.
