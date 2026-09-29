"""
data_processing.py  (Sentinel-2 NDVI version — rice growth stages)
==================================================================
Same role as the planting-method data_processing.py, but the datacube is
Sentinel-2 L2A NDVI instead of Sentinel-1 VH.

  * S2 L2A (B04, B08, SCL) -> Microsoft Planetary Computer `sentinel-2-l2a`
                              (STAC/COG, anonymous SAS signing, no GEE quota)
  * cloud/shadow mask      -> Scene Classification Layer (SCL)
  * radiometry             -> processing-baseline >= 04.00 BOA offset removed
                              (-1000 DN), so NDVI is consistent across years
  * compositing            -> regular INTERVAL_DAYS bins (max-value composite
                              by default); gaps stay NaN and are filled by
                              phenology.py, which tracks how many were real
  * cropland mask          -> ESA WorldCover class 40 from MPC (unchanged)

Output contract (consumed by phenology.run_phenology):
    ds[NDVI_BAND]  dims (time, y, x), float32 NDVI, NaN = cloud/gap/non-crop
    ds.attrs['spatial_dims'] = ['y', 'x']
    ds.attrs['window_start'], ds.attrs['window_end']  (ISO dates)

Memory: S2 is read EAGERLY per tile (keeps the SAS token fresh, same reason
as the RTC loader). With ~40-50 scenes in an 8-month window a 0.1 deg tile
(~1000 x 1000 px at 0.0001 deg) is ~0.3 GB raw; 0.25 deg tiles are ~2 GB, so
prefer TILE_DEG = 0.1 on standard Colab.
==================================================================
"""

import numpy as np
import pandas as pd
import xarray as xr

try:                                    # progress bars (Colab/Jupyter-friendly)
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, *a, **k):
        return iterable if iterable is not None else []

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
NDVI_BAND        = "NDVI"
GRID_SCALE_DEG   = 0.0001        # ~11 m; same grid as the planting-method product
INTERVAL_DAYS    = 10            # composite step (S2 revisit is 5 d; 10 d survives PH cloud)
COMPOSITE_METHOD = "max"         # 'max' (MVC, suppresses residual cloud) or 'median'
MAX_SCENE_CLOUD  = 80            # skip S2 items with eo:cloud_cover above this (%)

# SCL classes kept as clear observations:
#   4 vegetation, 5 not-vegetated (bare/stubble), 6 water (FLOODED PADDIES at
#   land prep / transplanting — must be kept or the planting trough disappears)
# dropped: 0 nodata, 1 saturated, 2 dark/topo shadow, 3 cloud shadow,
#          7 unclassified, 8/9 cloud med/high, 10 cirrus, 11 snow
SCL_CLEAR_CLASSES = (4, 5, 6)

# Processing baseline 04.00 (from 2022-01-25) adds BOA_ADD_OFFSET = -1000 DN.
# MPC serves the ESA values as-is, so remove it for consistency across years.
S2_OFFSET_DATE = pd.Timestamp("2022-01-25")
S2_BOA_OFFSET  = 1000
S2_SCALE       = 10000.0

MPC_STAC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"


def _open_catalog(retries=4):
    """MPC STAC client with retry/backoff (same as the S1 module)."""
    import time as _t
    import planetary_computer as pc
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3 import Retry

    retry = Retry(total=5, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    last = None
    for i in range(retries):
        try:
            return pystac_client.Client.open(
                MPC_STAC_URL, modifier=pc.sign_inplace,
                stac_io=StacApiIO(max_retries=retry))
        except Exception as e:
            last = e
            if i < retries - 1:
                _t.sleep(2 ** i)
    raise last


def _normalize_yx(ds):
    """odc/xee name geographic axes longitude/latitude, lon/lat or X/Y -> y/x."""
    names = set(ds.dims) | set(getattr(ds, "coords", {}))
    rename = {}
    for cand in ("longitude", "lon", "X"):
        if cand in names:
            rename[cand] = "x"; break
    for cand in ("latitude", "lat", "Y"):
        if cand in names:
            rename[cand] = "y"; break
    return ds.rename(rename) if rename else ds


# ==================================================================
# 1. Season window + regular anchor dates
# ==================================================================
def season_window(planting_month, year):
    """Same window as the planting-method pipeline: 1 month before the
    planting month through 6 months after (end of month). ~8 months, which
    covers the pre-planting fallow trough, the whole ~110-130 d rice cycle and
    the post-harvest trough."""
    import calendar
    start_month, start_year = planting_month - 1, year
    if start_month < 1:
        start_month += 12; start_year -= 1
    end_month, end_year = planting_month + 6, year
    if end_month > 12:
        end_month -= 12; end_year += 1
    last = calendar.monthrange(end_year, end_month)[1]
    return (f"{start_year}-{start_month:02d}-01",
            f"{end_year}-{end_month:02d}-{last:02d}")


planting_window = season_window          # alias: same name as the S1 module


def anchor_dates(start_date, end_date, interval_days=INTERVAL_DAYS):
    d0, d1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    n = int((d1 - d0).days // interval_days) + 1
    return pd.DatetimeIndex([d0 + pd.Timedelta(days=interval_days * k) for k in range(n)])


# ==================================================================
# 2. Per-scene NDVI from raw L2A DN (pure NumPy, testable offline)
# ==================================================================
def ndvi_from_l2a(red_dn, nir_dn, scl, acq_time,
                  clear_classes=SCL_CLEAR_CLASSES):
    """
    red_dn, nir_dn : uint16 DN arrays (0 = nodata)
    scl            : SCL class array (same shape)
    acq_time       : acquisition date (decides whether the BOA offset applies)
    Returns float32 NDVI with NaN for nodata / cloud / shadow.
    """
    red = np.asarray(red_dn, dtype=np.float32)
    nir = np.asarray(nir_dn, dtype=np.float32)
    valid = (red > 0) & (nir > 0) & np.isin(np.asarray(scl), clear_classes)

    off = S2_BOA_OFFSET if pd.Timestamp(acq_time) >= S2_OFFSET_DATE else 0
    red = np.clip((red - off) / S2_SCALE, 1e-4, None)   # negative BOA over water -> tiny +
    nir = np.clip((nir - off) / S2_SCALE, 1e-4, None)

    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    ndvi = np.where(valid & np.isfinite(ndvi), np.clip(ndvi, -1.0, 1.0), np.nan)
    return ndvi.astype(np.float32)


# ==================================================================
# 3. MPC Sentinel-2 L2A loader  (STAC -> xarray -> per-scene NDVI)
# ==================================================================
def load_s2_ndvi_stack(bbox, start_date, end_date, resolution=GRID_SCALE_DEG,
                       max_cloud=MAX_SCENE_CLOUD, pool=8):
    """
    Per-scene NDVI cube (time, y, x) float32 on an EPSG:4326 grid at
    `resolution`, cloud-masked with SCL. None if no scenes.
    Anonymous MPC access (pc.sign_inplace) — no subscription key.
    """
    import odc.stac
    import time as _t
    from rasterio.errors import RasterioIOError, WarpOperationError

    def _search():
        catalog = _open_catalog()                       # fresh signing
        return catalog.search(
            collections=[S2_COLLECTION], bbox=list(bbox),
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        ).item_collection()

    ds = None
    for attempt in range(4):
        try:
            items = _search()
            if len(items) == 0:
                if attempt < 3:
                    _t.sleep(2 ** attempt); continue
                return None
            ds = odc.stac.load(
                items, bands=["B04", "B08", "SCL"], bbox=list(bbox),
                crs="EPSG:4326", resolution=resolution,
                groupby="solar_day",                   # fuse same-day granules
                resampling="nearest",
                chunks=None,                           # EAGER: token still fresh
                pool=pool,
            )
            break
        except (RasterioIOError, WarpOperationError) as e:
            if attempt < 3:
                print(f"    S2 read retry {attempt+1}/3 ({e.__class__.__name__}) — re-signing")
                _t.sleep(2 ** attempt); continue
            raise

    ds = _normalize_yx(ds)
    times = pd.DatetimeIndex(ds["time"].values)
    ndvi = np.empty((len(times), ds.sizes["y"], ds.sizes["x"]), np.float32)
    for k, t in enumerate(times):
        ndvi[k] = ndvi_from_l2a(ds["B04"].values[k], ds["B08"].values[k],
                                ds["SCL"].values[k], t)
    out = xr.DataArray(ndvi, dims=("time", "y", "x"),
                       coords={"time": times, "y": ds["y"].values, "x": ds["x"].values},
                       name=NDVI_BAND)
    del ds
    return out


# ==================================================================
# 4. Temporal compositing to regular anchor dates
# ==================================================================
def composite_to_anchors(da_scene, start_date, end_date,
                         interval_days=INTERVAL_DAYS, method=COMPOSITE_METHOD):
    """
    Bin scenes into [anchor - dt/2, anchor + dt/2) and reduce with max (MVC)
    or median, ignoring NaN. Empty bins stay NaN — gap filling happens in
    phenology.py so it can count real observations and flag long gaps.
    """
    anchors = anchor_dates(start_date, end_date, interval_days)
    half = pd.Timedelta(days=interval_days / 2)
    t = pd.DatetimeIndex(da_scene["time"].values)
    arr = da_scene.values
    out = np.full((len(anchors),) + arr.shape[1:], np.nan, np.float32)
    reducer = np.nanmax if method == "max" else np.nanmedian
    import warnings
    for k, a in enumerate(anchors):
        sel = (t >= a - half) & (t < a + half)
        if sel.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices
                out[k] = reducer(arr[sel], axis=0)
    comp = xr.DataArray(out, dims=("time", "y", "x"),
                        coords={"time": anchors, "y": da_scene["y"].values,
                                "x": da_scene["x"].values}, name=NDVI_BAND)
    return comp


# ==================================================================
# 5. Cropland mask — ESA WorldCover class 40 from MPC (unchanged)
# ==================================================================
WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_CROPLAND_CLASS = 40


def cropland_mask_on_grid(target_da, cropland_class=WORLDCOVER_CROPLAND_CLASS):
    """Boolean (y, x) mask aligned to target_da (True = cropland)."""
    import odc.stac
    import time as _t

    y = target_da["y"].values
    x = target_da["x"].values
    res = GRID_SCALE_DEG
    bbox = [float(x.min()) - res / 2, float(y.min()) - res / 2,
            float(x.max()) + res / 2, float(y.max()) + res / 2]

    items = []
    for i in range(4):
        cat = _open_catalog()
        items = cat.search(collections=[WORLDCOVER_COLLECTION],
                           bbox=bbox).item_collection()
        if len(items) > 0:
            break
        if i < 3:
            _t.sleep(2 ** i)
    if len(items) == 0:
        raise RuntimeError(
            f"ESA WorldCover returned no items for bbox {bbox} after retries "
            f"(transient MPC STAC issue) — failing so skip_existing can retry.")

    lc = odc.stac.load(items, bands=["map"], bbox=bbox,
                       crs="EPSG:4326", resolution=res, chunks=None)
    lc = _normalize_yx(lc)["map"]
    if "time" in lc.dims:
        lc = lc.isel(time=0, drop=True) if lc.sizes["time"] == 1 \
            else lc.sortby("time").isel(time=-1, drop=True)
    mask = (lc == cropland_class)
    return mask.reindex_like(target_da, method="nearest",
                             tolerance=res / 2).fillna(False)


# ==================================================================
# 6. Glue: one tile/province -> NDVI datacube
# ==================================================================
def build_province_datacube(bbox, planting_month, year, apply_cropland_mask=True):
    """
    bbox : (west, south, east, north) EPSG:4326
    Returns Dataset {NDVI (time, y, x)} ready for phenology.run_phenology,
    or None if no S2 coverage for the window.
    """
    start, end = season_window(planting_month, year)
    scenes = load_s2_ndvi_stack(bbox, start, end)
    if scenes is None or scenes.sizes.get("time", 0) == 0:
        print(f"  no S2 L2A coverage for {start}..{end} — skipping")
        return None

    cube = composite_to_anchors(scenes, start, end)
    del scenes

    if apply_cropland_mask:
        mask = cropland_mask_on_grid(cube.isel(time=0, drop=True))
        cube = cube.where(mask)

    ds = cube.astype("float32").to_dataset(name=NDVI_BAND)
    ds.attrs.update({"spatial_dims": ["y", "x"],
                     "window_start": start, "window_end": end,
                     "interval_days": INTERVAL_DAYS})
    return ds


# ==================================================================
# 7. Tiled province builder (memory-bounded)
# ==================================================================
def generate_tiles(bbox, tile_deg=0.1):
    w, s, e, n = bbox
    nx = max(1, int(np.ceil((e - w) / tile_deg - 1e-9)))
    ny = max(1, int(np.ceil((n - s) / tile_deg - 1e-9)))
    tiles = []
    for ix in range(nx):
        x0 = w + ix * tile_deg
        for iy in range(ny):
            y0 = s + iy * tile_deg
            tiles.append((x0, y0, min(x0 + tile_deg, e), min(y0 + tile_deg, n)))
    return tiles


def _province_grid_coords(bbox, res=GRID_SCALE_DEG):
    w, s, e, n = bbox
    xs = np.arange(w + res / 2, e, res)
    ys = np.arange(n - res / 2, s, -res)
    return ys, xs


def _mosaic_tiles(pairs, bbox, res=GRID_SCALE_DEG):
    """
    Merge per-tile Datasets onto ONE province grid. Every data_var is
    mosaicked (2-D (y,x) products or 3-D (time,y,x) cubes). Each province
    pixel belongs to exactly one tile by its centre (half-open bins).
    """
    pairs = [(d, tb) for d, tb in pairs if d is not None]
    if not pairs:
        return None
    ys, xs = _province_grid_coords(bbox, res)
    ref = pairs[0][0]
    out = {}
    for v in ref.data_vars:
        extra = [d for d in ref[v].dims if d not in ("y", "x")]
        shape = tuple(ref.sizes[d] for d in extra) + (ys.size, xs.size)
        coords = {d: ref[d].values for d in extra if d in ref.coords}
        coords.update({"y": ys, "x": xs})
        out[v] = xr.DataArray(np.full(shape, np.nan, np.float32),
                              dims=tuple(extra) + ("y", "x"), coords=coords)
    for d, tb in pairs:
        w, s, e, n = tb
        sel_x = xs[(xs >= w) & (xs < e)]
        sel_y = ys[(ys > s) & (ys <= n)]
        if sel_x.size == 0 or sel_y.size == 0:
            continue
        t = d.reindex(y=sel_y, x=sel_x, method="nearest", tolerance=res / 2)
        for v in out:
            out[v].loc[dict(y=sel_y, x=sel_x)] = t[v].astype("float32").values
    ds = xr.Dataset(out)
    ds.attrs.update({k: v for k, v in ref.attrs.items()})
    ds.attrs["spatial_dims"] = ["y", "x"]
    return ds


def build_province_datacube_tiled(bbox, planting_month, year, tile_deg=0.1,
                                  per_tile_fn=None, apply_cropland_mask=True):
    """
    Tiled province builder.

    per_tile_fn : None  -> returns the full NDVI cube mosaic (time, y, x).
                           Fine for small AOIs; heavy for large provinces.
                  fn(ds) -> applied to each tile cube right after it is built
                           (e.g. phenology.run_phenology). Only its (small, 2-D)
                           outputs are kept and mosaicked, so peak memory is one
                           tile cube. Phenology is per-pixel in time, so tiling
                           introduces no seams.
    """
    tiles = generate_tiles(bbox, tile_deg)
    print(f"  province split into {len(tiles)} tile(s) of {tile_deg} deg")
    import gc
    pairs, attrs = [], {}
    for tb in tqdm(tiles, desc="tiles", unit="tile"):
        ds_t = build_province_datacube(tb, planting_month, year,
                                       apply_cropland_mask=apply_cropland_mask)
        if ds_t is None:
            continue
        attrs = dict(ds_t.attrs)
        if per_tile_fn is not None:
            ds_t = per_tile_fn(ds_t)
        pairs.append((ds_t, tb))
        gc.collect()
    ds = _mosaic_tiles(pairs, bbox)
    if ds is not None:
        for k, v in attrs.items():
            ds.attrs.setdefault(k, v)
    return ds


# ==================================================================
# 8. Clip to a vector boundary (all 2-D variables at once)
# ==================================================================
def _as_geometry_list(geometry):
    try:
        import geopandas as gpd
        if isinstance(geometry, gpd.GeoDataFrame):
            return list(geometry.geometry.values)
        if isinstance(geometry, gpd.GeoSeries):
            return list(geometry.values)
    except Exception:
        pass
    from shapely.geometry.base import BaseGeometry
    if isinstance(geometry, BaseGeometry):
        return [geometry]
    if isinstance(geometry, (list, tuple)):
        return list(geometry)
    return [geometry]


def clip_to_geometry(ds, geometry, drop=True, all_touched=True):
    """
    Set every (y, x) variable to NaN outside the polygon(s); with drop=True
    also trim the extent to the polygon's bounding box. Geometry must be
    EPSG:4326. Works before int conversion (NaN nodata), so one call clips
    all date/metric bands consistently.
    """
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    geoms = _as_geometry_list(geometry)
    ys, xs = ds["y"].values, ds["x"].values
    if ys[0] < ys[-1]:
        ds = ds.sortby("y", ascending=False); ys = ds["y"].values
    res_x = float(abs(xs[1] - xs[0])) if xs.size > 1 else GRID_SCALE_DEG
    res_y = float(abs(ys[1] - ys[0])) if ys.size > 1 else GRID_SCALE_DEG
    transform = from_origin(xs[0] - res_x / 2, ys[0] + res_y / 2, res_x, res_y)
    inside = ~geometry_mask(geoms, out_shape=(ys.size, xs.size),
                            transform=transform, all_touched=all_touched)
    inside_da = xr.DataArray(inside, dims=("y", "x"), coords={"y": ys, "x": xs})

    out = ds.copy()
    for v in out.data_vars:
        if "y" in out[v].dims and "x" in out[v].dims:
            out[v] = out[v].where(inside_da)
    if drop:
        rows = np.where(inside.any(axis=1))[0]
        cols = np.where(inside.any(axis=0))[0]
        if rows.size and cols.size:
            out = out.isel(y=slice(rows[0], rows[-1] + 1),
                           x=slice(cols[0], cols[-1] + 1))
    out.attrs.update(ds.attrs)
    out.attrs["spatial_dims"] = ["y", "x"]
    return out
