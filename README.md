# ai-cropgrowth: rice growth stages from Sentinel-2 NDVI

Maps rice growth stages in the Philippines from Sentinel-2 NDVI time series.
It uses formula-based phenology (thresholds, Savitzky-Golay smoothing and NDVI
derivatives), so there's no model to train. Outputs are Cloud-Optimized
GeoTIFFs on a **30 m** grid by default (10 m optional), written to Google Cloud
Storage.

## Data sources

Set `DATA_SOURCE` in the notebook (`dp.set_data_source`).

| | **`'hls'`**: Harmonized Landsat Sentinel-2 v2.0 | **`'s2'`**: Sentinel-2 L2A |
|---|---|---|
| Sensors | Landsat 8/9 (`hls2-l30`) + Sentinel-2 (`hls2-s30`) | Sentinel-2 |
| Revisit | ~2–3 days combined | 5 days |
| Native resolution | 30 m | 10 m (red/NIR) |
| Cloud mask | Fmask: drops cirrus, cloud, adjacent cloud, shadow, snow and high aerosol; keeps water (flooded paddies) | SCL classes 4/5/6 |
| Calibration | Landsat and Sentinel-2 brought to one spectral response and BRDF-normalised | ESA processing-baseline offset removed |
| NIR band | B8A (S30) / B05 (L30) | B08 |

- **HLS** gives more clear observations in cloudy seasons, which is the main
  limit for rice phenology in the Philippines. It is the default in the
  notebook, on the 30 m grid.
- **S2** is the choice for a 10 m grid. It may also be a few days fresher in
  recent mode, because HLS is produced by NASA after the source scenes arrive.
- **Re-check thresholds when switching sources.** HLS NDVI uses the narrow NIR
  band, so absolute values differ a little from S2. Confirm `MIN_PEAK_NDVI` and
  `MAX_BASE_NDVI` with the quick check.
- **Searches**: Planetary Computer accepts only one collection per search, so
  HLS makes two searches (L30, S30) and merges them. The two sensors are loaded
  separately because their NIR bands have different names, then combined in
  time.

## Run modes

| | **Periodic** | **Recent** |
|---|---|---|
| Areas | nationwide, regional or provincial (`AREA_LEVEL`) | municipal or barangay (`AOI_PATH`, `AOI_NAMES`) |
| When | on a schedule: `CADENCE = 'monthly'` or `'semimonthly'` (1st–15th, 16th–end) | on demand (near real time) |
| Time window | each province's season (planting month −1 … +6), with data up to `AS_OF` | rolling `RECENT_LOOKBACK_DAYS` (240) ending today |
| Crop cycle | the season's cycle (highest NDVI peak) | the **most recent** cycle (`PEAK_SELECT='last'`) |
| Output | one stage map per province per period | stage **today** + days since the newest clear observation + hectares per stage |
| Code | `pipeline.run_periodic_unit`, `pipeline.stage_maps` | `pipeline.run_recent`, `pipeline.stage_area_summary` |

Set `MODE` in the notebook's config cell, then run the **Periodic run** or
**Recent run** section.

- **Periodic, `PERIODS = 'last_complete'`** (the default): produces the
  newest period that has ended. A monthly run on 1 March produces February; a
  semimonthly run on the 16th produces the 1st–15th.
  - `'current'` produces the period still in progress (provisional).
  - `'season'` backfills every period of the season that has ended.
  - Provinces whose season doesn't include the period are reported as
    `out_of_season`.
- **Recent**: composites are anchored on the run date, so the newest scenes
  count.
  - The run follows the latest crop cycle, so a young crop isn't mistaken for
    the previous harvested one.
  - `data_age_days` shows how stale each pixel's stage is. Sentinel-2 revisits
    every 5 days, but clouds can hide a field for weeks.

```
HLS v2 or S2 L2A (Microsoft Planetary Computer) → Fmask / SCL cloud mask → 10-day NDVI composites
→ ESA WorldCover cropland mask → SG smoothing → 15 % amplitude threshold
→ NDVI derivatives / peaks → transition dates → monthly stage maps
```

## Repository layout

| Path | Role |
|---|---|
| `apps/data_processing.py` | Season window, S2 NDVI loading from MPC, compositing, cropland mask, tiling/mosaicking, clipping to the province boundary |
| `apps/phenology.py` | Per-pixel phenology (transition dates + QC), stage classification (date, month or any period), spatial clean-up, COG export |
| `apps/pipeline.py` | The two run modes: periods, admin-unit selection, periodic and recent runs, area summaries |
| `run_growth_stages.ipynb` | Colab driver: config, GCS auth, a quick-check AOI with plots, the periodic batch and the recent run |

## Setup (Google Colab)

1. Put `apps/` (all three `.py` files) on Google Drive and point the
   `sys.path.insert(...)` line in the notebook's config cell at it.
2. Run the install cell:
   `pip install pystac-client odc-stac planetary-computer xarray rioxarray scipy geopandas matplotlib google-cloud-storage`.
3. Authenticate to GCS (the `gcloud auth` + impersonation cells).
4. Set `MODE`, then that mode's settings in the config cell: `AREA_LEVEL`,
   `CADENCE`, `YEAR`, `VECTOR_PATH`, ... for periodic, or `AOI_PATH`,
   `AOI_NAME_COL`, `AOI_NAMES` for recent.
5. Run the **Quick check** cells on a small rice area and tune `PHENO_CFG`.
   Then run the section for your mode.

Sentinel-2 and WorldCover are read anonymously from Planetary Computer, so no
API key is needed.

## Configuration

- **Grid**: `RESOLUTION_M = 30` (default) or `10`, applied with
  `dp.set_resolution()`.
  - At 30 m, red/NIR are averaged from the 10 m bands, using the COG
    overviews, so about 9× fewer bytes are read.
  - The SCL cloud mask and the WorldCover classes take the majority class.
  - `TILE_DEG` follows the grid: 0.25° at 30 m, 0.1° at 10 m. Either way a tile
    is about 1000×1000 px in memory.
- **Notebook**:
  - `TILE_WORKERS`: tiles in parallel; ~1–1.5 GB RAM each, 2–3 on standard
    Colab.
  - `STAGE_METHOD`: `dominant` (stage that fills most of the period) or
    `midpoint`.
  - Also `DATE_MEDIAN_RADIUS`, `SAVE_PHENOLOGY_BANDS` and `TILE_CACHE_DIR`.
- **Admin level**: `select_units` selects provinces. `AREA_LEVEL = 'regional'`
  needs a region column (`REGION_COL`) in `VECTOR_PATH`; set it to the column
  your file actually has. Processing and outputs are always per province.
- **`data_processing` module settings** can be changed from the notebook
  (e.g. `dp.INTERVAL_DAYS = 10`). They are read when each function runs:
  `INTERVAL_DAYS`, `COMPOSITE_METHOD`, `MAX_SCENE_CLOUD`, `GRID_SCALE_DEG`,
  `SCL_CLEAR_CLASSES`.
- **Phenology**: pass overrides to `ph.run_phenology(ds, **PHENO_CFG)`. See
  `phenology.DEFAULTS` for every parameter (threshold fraction, SG window,
  rice-likeness QC limits, search windows, …).

## Outputs

**Periodic**: `{COG_LABEL}_{PROVINCE}_{YYYYMM}.tiff` (monthly) or
`…_{YYYYMM}H1.tiff` / `…H2.tiff` (semimonthly). Each is a single-band int16
stage map for one period:

  | Value | Stage |
  |---|---|
  | −1 | nodata (non-cropland / outside province) |
  | 0 | no rice cycle detected |
  | 1 | pre-planting / transplanting |
  | 2 | vegetative (emergence → panicle initiation) |
  | 3 | reproductive (panicle initiation → peak NDVI) |
  | 4 | ripening / maturity (peak → harvest) |
  | 5 | harvested / post-harvest |

**Recent**: in `RECENT_PREFIX`:
- `{AREA}_{YYYYMMDD}.tiff`: 3 int16 bands: `growth_stage` (classes above),
  `data_age_days` and `qc`.
- `{AREA}_{YYYYMMDD}_summary.csv`: pixels, hectares and % of mapped cropland
  per stage.

**QA (optional)**:
- `qa/{COG_LABEL}_{PROVINCE}_phenology_dates.tiff` (when
  `SAVE_PHENOLOGY_BANDS = True`): 15 int16 bands. They hold 8 transition dates
  (plant, emergence, tillering, panicle_init, heading, peak, maturity, harvest)
  as day-of-year relative to Jan 1 of `YEAR`, plus season length, NDVI
  min/max/NDVI₁₅ (×10000), QC, number of clear observations and longest gap.
  Stage maps for any other date can be rebuilt from this file with no
  reprocessing (see the last notebook cell).

QC codes: `0` complete cycle, `1` ongoing past peak, `2` ongoing before
peak; `10` too few clear observations, `11` no interior peak, `12` amplitude
too small, `13` peak NDVI too low, `14` base NDVI too high, `15` no rising
15 % crossing, `16` season length implausible.

## Performance

- **One catalog search per province.** Sentinel-2 and WorldCover are each
  searched once, and every tile filters those results locally. Asset URLs are
  signed when each file is read (`patch_url=planetary_computer.sign`), so the
  results stay valid for a multi-hour province run.
- **Cropland first.** Each tile reads the one-band WorldCover map before any
  Sentinel-2 data. Tiles with no cropland are skipped, and Sentinel-2 is read
  only over the bounding box of the tile's cropland.
- **Only tiles inside the province outline** are processed.
- **Parallel tiles.** Use `tile_workers` / `TILE_WORKERS`.
- **One GCS listing.** The batch run checks for existing outputs with a single
  listing of `GCS_PREFIX` instead of one request per month.

## How failures are handled

- **Tiles outside the province outline** (e.g. open sea inside a coastal
  province's bounding box) are dropped before processing (`geometry=`).
- **Offshore tiles**: a tile with no WorldCover item is skipped before any
  Sentinel-2 data is downloaded.
- **Unreadable Sentinel-2 scenes**: some Planetary Computer files open without
  a coordinate system. odc then fails with `AssertionError: src.crs is not
  None`. This isn't covered by `fail_on_error`, and the same scene fails on
  every retry.
  - On a failed read, the header of each scene's band files is checked, and
    the unusable scenes are dropped and logged (`dropping N unreadable S2
    scene(s): <id> [bands]`).
  - The tile is then reloaded without them.
  - Bad scenes are remembered, so later tiles skip them without checking
    again.
- **Other read failures** are retried with freshly signed URLs.
- **Empty province search**: if the province-wide Sentinel-2 search returns
  nothing after its retries, the province is reported as failed rather than
  "no coverage", because that is almost always a temporary Planetary Computer
  problem.
- **Per-tile retries**: each tile gets `tile_retries` extra attempts. If a tile
  still fails, the province is reported as failed so `skip_existing` retries it
  on the next batch run. With `cache_dir` set, finished tiles are saved to disk
  and reused on that rerun, so a large province doesn't start over. Offshore
  and no-cropland tiles are cached as skip markers, so a rerun doesn't
  re-check them.
  The cache key covers the time window, grid size, compositing, cloud
  settings and `PHENO_CFG`. Changing any of them, or running a new period,
  never reuses stale tiles.
- To resume the batch after a Colab disconnect, run
  `batch_all(provinces, start_from='PROVINCE')`.

## Method reference

The detailed rules for each transition date (NDVI₁₅ crossings, maximum growth
rate, maximum acceleration, steepest decline, …) are documented in the
`phenology.py` module docstring. Data-source and memory notes are in the
`data_processing.py` docstring.
