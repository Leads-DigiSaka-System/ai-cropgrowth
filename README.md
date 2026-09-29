# ai-cropgrowth: rice growth stages from Sentinel-2 NDVI

Maps rice growth stages across Philippine provinces from Sentinel-2 NDVI time
series. It uses formula-based phenology (thresholds, Savitzky-Golay smoothing
and NDVI derivatives), so there's no model to train. The output is one
growth-stage map per month per province, written as Cloud-Optimized GeoTIFFs
to Google Cloud Storage.

```
S2 L2A (Microsoft Planetary Computer) → SCL cloud mask → 10-day NDVI composites
→ ESA WorldCover cropland mask → SG smoothing → 15 % amplitude threshold
→ NDVI derivatives / peaks → transition dates → monthly stage maps
```

## Repository layout

| Path | Role |
|---|---|
| `apps/data_processing.py` | Season window, S2 NDVI loading from MPC, compositing, cropland mask, tiling/mosaicking, clipping to the province boundary |
| `apps/phenology.py` | Per-pixel phenology (transition dates + QC), stage classification, spatial clean-up, COG export |
| `run_growth_stages.ipynb` | Colab driver: config, GCS auth, a quick-check AOI with plots, and the batch run over all provinces |

## Setup (Google Colab)

1. Put `apps/` on Google Drive and point the `sys.path.insert(...)` line in the
   notebook's config cell at it.
2. Run the install cell:
   `pip install pystac-client odc-stac planetary-computer xarray rioxarray scipy geopandas matplotlib google-cloud-storage`.
3. Authenticate to GCS (the `gcloud auth` + impersonation cells).
4. Set `YEAR`, `GCS_PREFIX`, `COG_LABEL`, `VECTOR_PATH` and `PLANT_MO_COL` in
   the config cell.
5. Run the **Quick check** cells on a small rice area, tune `PHENO_CFG`, then
   run **Batch Run**.

Sentinel-2 and WorldCover are read anonymously from Planetary Computer, so no
API key is needed.

## Configuration

- **Notebook**: `TILE_DEG` (0.1° keeps one tile ≈ 1000×1000 px in memory),
  `YEAR`, `MONTHS`, `STAGE_METHOD` (`dominant` / `midmonth`),
  `DATE_MEDIAN_RADIUS`, `SAVE_PHENOLOGY_BANDS`, `TILE_CACHE_DIR`.
- **`data_processing` module settings** can be changed from the notebook
  (e.g. `dp.INTERVAL_DAYS = 10`). They are read when each function runs:
  `INTERVAL_DAYS`, `COMPOSITE_METHOD`, `MAX_SCENE_CLOUD`, `GRID_SCALE_DEG`,
  `SCL_CLEAR_CLASSES`.
- **Phenology**: pass overrides to `ph.run_phenology(ds, **PHENO_CFG)`. See
  `phenology.DEFAULTS` for every parameter (threshold fraction, SG window,
  rice-likeness QC limits, search windows, …).

## Outputs

- `{COG_LABEL}_{PROVINCE}_{YYYYMM}.tiff`: single-band int16 stage map for each
  month of the season window (planting month −1 … +6).

  | Value | Stage |
  |---|---|
  | −1 | nodata (non-cropland / outside province) |
  | 0 | no rice cycle detected |
  | 1 | pre-planting / transplanting |
  | 2 | vegetative (emergence → panicle initiation) |
  | 3 | reproductive (panicle initiation → peak NDVI) |
  | 4 | ripening / maturity (peak → harvest) |
  | 5 | harvested / post-harvest |

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

## How failures are handled

- **Tiles outside the province outline** (e.g. open sea inside a coastal
  province's bounding box) are dropped before processing (`geometry=`).
- **Offshore tiles**: before any Sentinel-2 data is downloaded, a cheap
  WorldCover search is run for the tile. If it finds no item, the tile is
  skipped.
- **Failed image reads**: an `odc` "src.crs is not None" assertion (usually an
  expired access token) triggers a fresh, re-signed catalog search and a retry.
- **Per-tile retries**: each tile gets `tile_retries` extra attempts. If a tile
  still fails, the province is reported as failed so `skip_existing` retries it
  on the next batch run. With `cache_dir` set, finished tiles are saved to disk
  and reused on that rerun, so a large province doesn't start over.
  **The cache key is year + planting month + tile bounds only.** Use a new
  `TILE_CACHE_DIR` (or empty it) after changing `PHENO_CFG` or `dp.*`
  settings.
- To resume the batch after a Colab disconnect, run
  `batch_all(provinces, start_from='PROVINCE')`.

## Method reference

The detailed rules for each transition date (NDVI₁₅ crossings, maximum growth
rate, maximum acceleration, steepest decline, …) are documented in the
`phenology.py` module docstring. Data-source and memory notes are in the
`data_processing.py` docstring.
