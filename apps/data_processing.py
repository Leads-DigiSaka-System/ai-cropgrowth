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

Cost: STAC is searched once per province; asset URLs are signed at read
time (odc.stac.load(patch_url=planetary_computer.sign)). Per tile, the cheap
WorldCover mask is read first; tiles with no cropland skip S2 entirely and
S2 is read only over the cropland extent.

Memory: S2 is read eagerly per tile. With ~40-50 scenes in an 8-month window
a 0.1 deg tile (~1000 x 1000 px at 0.0001 deg) is ~0.3 GB raw; 0.25 deg tiles
are ~2 GB, so prefer TILE_DEG = 0.1 on standard Colab. Peak memory scales with
tile_workers.
==================================================================
"""

import threading

import numpy as np
import pandas as pd
import xarray as xr

try:                                    # progress bars (Colab/Jupyter-friendly)
    from tqdm.auto import tqdm
except Exception:
    class tqdm:                                         # minimal no-op stand-in
        def __init__(self, iterable=None, *a, **k):
            self.iterable = iterable if iterable is not None else []
        def __iter__(self):
            return iter(self.iterable)
        def update(self, n=1):
            pass
        def close(self):
            pass

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
NDVI_BAND        = "NDVI"
M_PER_DEG        = 111_320.0     # metres per degree (latitude; longitude at the equator)
RESOLUTION_M     = 30            # output grid size in metres — change with set_resolution()
GRID_SCALE_DEG   = RESOLUTION_M / M_PER_DEG   # ~0.00027 deg at 30 m (0.0001 deg ~ 11 m)
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



def set_resolution(meters):
    """Set the output grid size (metres; e.g. 30 or 10). Coarser than ~15 m,
    reflectance is AVERAGED from the 10 m bands (reading the COG overviews, so
    far fewer bytes) and the SCL / WorldCover classes take the majority class.
    Returns the grid step in degrees."""
    global RESOLUTION_M, GRID_SCALE_DEG
    RESOLUTION_M = float(meters)
    GRID_SCALE_DEG = RESOLUTION_M / M_PER_DEG
    return GRID_SCALE_DEG


def _coarse():
    return GRID_SCALE_DEG * M_PER_DEG > 15


def _s2_resampling():
    # SCL by majority, not average: a class code can't be averaged
    return ({"B04": "average", "B08": "average", "SCL": "mode"} if _coarse()
            else "nearest")


MPC_STAC_URL  = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"


_CATALOG = None


def _open_catalog(retries=4, fresh=False):
    """MPC STAC client with retry/backoff, reused across calls (fresh=True
    reopens it). Items come back UNSIGNED: asset URLs are signed at read time
    via odc.stac.load(patch_url=planetary_computer.sign), which refreshes the
    SAS token as needed, so one province-wide search stays usable for hours."""
    global _CATALOG
    if _CATALOG is not None and not fresh:
        return _CATALOG
    import time as _t
    import pystac_client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3 import Retry

    retry = Retry(total=5, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    last = None
    for i in range(retries):
        try:
            _CATALOG = pystac_client.Client.open(
                MPC_STAC_URL, stac_io=StacApiIO(max_retries=retry))
            return _CATALOG
        except Exception as e:
            last = e
            if i < retries - 1:
                _t.sleep(2 ** i)
    raise last


def _sign_url(href):
    """patch_url for odc.stac.load: sign each asset URL when it is read."""
    import planetary_computer as pc
    return pc.sign(href)


def _search(retries=4, **query):
    """STAC search -> list of items, retrying errors and empty results
    (MPC occasionally returns an empty page transiently). [] if still empty."""
    import time as _t
    items = []
    for i in range(retries):
        try:
            items = list(_open_catalog(fresh=i > 0).search(**query).items())
            if items:
                return items
        except Exception:
            if i == retries - 1:
                raise
        if i < retries - 1:
            _t.sleep(2 ** i)
    return items


def search_s2_items(bbox, start_date, end_date, max_cloud=None, retries=4):
    """Sentinel-2 L2A items over bbox in [start_date, end_date] (unsigned)."""
    max_cloud = MAX_SCENE_CLOUD if max_cloud is None else max_cloud
    return _search(retries, collections=[S2_COLLECTION], bbox=list(bbox),
                   datetime=f"{start_date}/{end_date}",
                   filter={"op": "<", "args": [{"property": "eo:cloud_cover"}, max_cloud]},
                   filter_lang="cql2-json")


def items_in_bbox(items, bbox):
    """Items whose footprint intersects bbox — lets one province-wide search
    serve every tile without another STAC request."""
    from shapely.geometry import box, shape
    b = box(*bbox)
    out = []
    for it in items:
        g = getattr(it, "_footprint", None)
        if g is None:
            g = shape(it.geometry) if it.geometry else box(*it.bbox)
            try:
                it._footprint = g                        # cache on the item
            except Exception:
                pass
        if g.intersects(b) and not g.touches(b):         # shared edge only != overlap
            out.append(it)
    return out


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


def anchor_dates(start_date, end_date, interval_days=None, align="start"):
    """Regular composite dates every interval_days. align='start' anchors on
    start_date (season products); align='end' anchors on end_date, so the last
    composite is centred on the newest data (recent / near-real-time products)."""
    interval_days = interval_days or INTERVAL_DAYS
    d0, d1 = pd.Timestamp(start_date), pd.Timestamp(end_date)
    n = int((d1 - d0).days // interval_days) + 1
    step = pd.Timedelta(days=interval_days)
    if align == "end":
        return pd.DatetimeIndex([d1 - step * k for k in range(n)][::-1])
    return pd.DatetimeIndex([d0 + step * k for k in range(n)])


# ==================================================================
# 2. Per-scene NDVI from raw L2A DN (pure NumPy, testable offline)
# ==================================================================
def ndvi_from_l2a(red_dn, nir_dn, scl, acq_time, clear_classes=None):
    """
    red_dn, nir_dn : uint16 DN arrays (0 = nodata)
    scl            : SCL class array (same shape)
    acq_time       : acquisition date (decides whether the BOA offset applies)
    Returns float32 NDVI with NaN for nodata / cloud / shadow.
    """
    clear_classes = SCL_CLEAR_CLASSES if clear_classes is None else clear_classes
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
S2_BANDS = ["B04", "B08", "SCL"]

# Asset hrefs (unsigned) found unreadable or without a CRS / geotransform.
# Shared across tiles: once a bad scene is found it is dropped everywhere.
_BAD_HREFS = set()
_BAD_LOCK = threading.Lock()


def _asset_is_bad(href):
    """Open one COG header and check it is georeferenced. True = unusable."""
    import rasterio
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(_sign_url(href)) as r:
                return r.crs is None or r.transform.is_identity
    except Exception:
        return True


def find_bad_s2_assets(items, bands=None, workers=16):
    """Probe every band of every item (header reads only) and return
    {item_id: [bad bands]}. Adds the bad hrefs to _BAD_HREFS."""
    from concurrent.futures import ThreadPoolExecutor
    bands = bands or S2_BANDS
    jobs = [(it.id, b, it.assets[b].href) for it in items
            for b in bands if b in getattr(it, "assets", {})]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        flags = list(ex.map(lambda j: _asset_is_bad(j[2]), jobs))
    bad = {}
    for (iid, b, href), f in zip(jobs, flags):
        if f:
            bad.setdefault(iid, []).append(b)
            with _BAD_LOCK:
                _BAD_HREFS.add(href)
    return bad


def _drop_bad(items, bands=None):
    bands = bands or S2_BANDS
    if not _BAD_HREFS:
        return list(items)
    return [it for it in items
            if not any(b in it.assets and it.assets[b].href in _BAD_HREFS for b in bands)]


def load_s2_ndvi_stack(bbox, start_date, end_date, resolution=None,
                       max_cloud=None, pool=8, items=None):
    """
    Per-scene NDVI cube (time, y, x) float32 on an EPSG:4326 grid at
    `resolution`, cloud-masked with SCL. None if no scenes.
    items : pre-searched S2 items (e.g. one province-wide search filtered with
            items_in_bbox); searched here if None.
    Anonymous MPC access (URLs signed at read time) — no subscription key.

    Bad scenes: some MPC S2 COGs open without a CRS, which odc reports as
    `AssertionError: src.crs is not None` (not covered by fail_on_error) and
    which fails on every retry. On a failed read every item's band headers
    are probed, the unusable scenes are dropped (and remembered for all later
    tiles), and the load is repeated without them.
    """
    import odc.stac
    import time as _t
    from rasterio.errors import RasterioError

    resolution = resolution or GRID_SCALE_DEG
    if items is None:
        items = search_s2_items(bbox, start_date, end_date, max_cloud)
    items = _drop_bad(items)
    if len(items) == 0:
        return None

    ds, probed = None, False
    for attempt in range(4):
        try:
            ds = odc.stac.load(
                items, bands=S2_BANDS, bbox=list(bbox),
                crs="EPSG:4326", resolution=resolution,
                groupby="solar_day",                   # fuse same-day granules
                resampling=_s2_resampling(),
                chunks=None,                           # eager, bounded by the tile size
                pool=pool,
                patch_url=_sign_url,                   # fresh SAS token per read
            )
            break
        except (RasterioError, AssertionError) as e:
            if attempt == 3:
                raise
            if not probed:                             # find the scene(s) at fault
                probed = True
                bad = find_bad_s2_assets(items)
                if bad:
                    print(f"    dropping {len(bad)} unreadable S2 scene(s): "
                          + ", ".join(f"{k} [{'/'.join(v)}]" for k, v in sorted(bad.items())))
                    items = _drop_bad(items)
                    if not items:
                        return None
                    continue                           # retry at once without them
            print(f"    S2 read retry {attempt+1}/3 ({e.__class__.__name__})")
            _t.sleep(2 ** attempt)

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
                         interval_days=None, method=None, align="start"):
    """
    Bin scenes into [anchor - dt/2, anchor + dt/2) and reduce with max (MVC)
    or median, ignoring NaN. Empty bins stay NaN — gap filling happens in
    phenology.py so it can count real observations and flag long gaps.
    """
    interval_days = interval_days or INTERVAL_DAYS
    method = method or COMPOSITE_METHOD
    anchors = anchor_dates(start_date, end_date, interval_days, align)
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
# 5. Cropland mask — ESA WorldCover class 40 from MPC
# ==================================================================
WORLDCOVER_COLLECTION = "esa-worldcover"
WORLDCOVER_CROPLAND_CLASS = 40


class NoWorldCoverError(RuntimeError):
    """ESA WorldCover has no item for the bbox (open sea / outside coverage)."""


def worldcover_items(bbox, retries=4):
    """WorldCover STAC items for bbox, with retry. Raises NoWorldCoverError if
    none come back — WorldCover has no tiles over open sea, so a tile that
    falls entirely offshore lands here and can be skipped."""
    items = _search(retries, collections=[WORLDCOVER_COLLECTION], bbox=list(bbox))
    if not items:
        raise NoWorldCoverError(f"ESA WorldCover returned no items for bbox {list(bbox)}")
    return items


def load_cropland_mask(bbox, items=None, cropland_class=WORLDCOVER_CROPLAND_CLASS):
    """Boolean (y, x) cropland mask for bbox on the GRID_SCALE_DEG grid.
    Cheap (one uint8 band, one date) — loaded BEFORE the S2 stack so tiles
    without cropland never download S2."""
    import odc.stac
    res = GRID_SCALE_DEG
    if items is None:
        items = worldcover_items(bbox)
    lc = odc.stac.load(items, bands=["map"], bbox=list(bbox), crs="EPSG:4326",
                       resolution=res, chunks=None, patch_url=_sign_url,
                       resampling="mode" if _coarse() else "nearest")
    lc = _normalize_yx(lc)["map"]
    if "time" in lc.dims:
        lc = lc.isel(time=0, drop=True) if lc.sizes["time"] == 1 \
            else lc.sortby("time").isel(time=-1, drop=True)
    return lc == cropland_class


def _mask_extent(mask, bbox):
    """bbox of the True pixels of a (y, x) mask, clipped to bbox."""
    res = GRID_SCALE_DEG
    ys = mask["y"].values[mask.any("x").values]
    xs = mask["x"].values[mask.any("y").values]
    w, s, e, n = bbox
    return (max(w, float(xs.min()) - res / 2), max(s, float(ys.min()) - res / 2),
            min(e, float(xs.max()) + res / 2), min(n, float(ys.max()) + res / 2))


def cropland_mask_on_grid(target_da, cropland_class=WORLDCOVER_CROPLAND_CLASS,
                          items=None, mask=None):
    """Boolean (y, x) mask aligned to target_da (True = cropland).
    mask  : a load_cropland_mask() result to reuse (otherwise loaded here)
    items : pre-fetched worldcover_items() (searched here if None)."""
    res = GRID_SCALE_DEG
    if mask is None:
        y = target_da["y"].values
        x = target_da["x"].values
        bbox = [float(x.min()) - res / 2, float(y.min()) - res / 2,
                float(x.max()) + res / 2, float(y.max()) + res / 2]
        mask = load_cropland_mask(bbox, items, cropland_class)
    return mask.reindex_like(target_da, method="nearest",
                             tolerance=res / 2).fillna(False).astype(bool)


# ==================================================================
# 6. Glue: one tile/province -> NDVI datacube
# ==================================================================
def _fmt_bbox(bbox):
    return tuple(round(float(v), 4) for v in bbox)


def build_province_datacube(bbox, planting_month, year, apply_cropland_mask=True,
                            s2_items=None, wc_items=None, window=None, align="start"):
    return _build_tile(bbox, planting_month, year, apply_cropland_mask,
                       s2_items, wc_items, window, align)[0]


# skip reasons that are fixed properties of the tile (safe to cache)
_PERMANENT_SKIPS = ("offshore", "no_cropland")


def _build_tile(bbox, planting_month, year, apply_cropland_mask=True,
                s2_items=None, wc_items=None, window=None, align="start"):
    """
    bbox : (west, south, east, north) EPSG:4326
    s2_items / wc_items : optional pre-searched STAC items (e.g. from one
        province-wide search); filtered to bbox here. Searched if None.
    window : (start, end) ISO dates overriding season_window(planting_month,
        year) — e.g. a season clipped at today, or a rolling recent window.
        planting_month / year are ignored when it is given.
    align  : composite anchoring, 'start' (season) or 'end' (recent).
    Returns Dataset {NDVI (time, y, x)} ready for phenology.run_phenology,
    or None if the tile has no WorldCover coverage (offshore), no cropland,
    or no S2 coverage for the window.

    Order matters for cost: the cheap cropland mask is loaded first, tiles
    without cropland are skipped, and S2 is read only over the cropland extent.

    (_build_tile returns (Dataset | None, skip reason | None).)
    """
    start, end = window or season_window(planting_month, year)
    load_bbox, crop = tuple(bbox), None
    if apply_cropland_mask:
        try:
            wc = worldcover_items(bbox) if wc_items is None else items_in_bbox(wc_items, bbox)
        except NoWorldCoverError:
            wc = []
        if not wc:
            print(f"  no WorldCover coverage for {_fmt_bbox(bbox)} (offshore) — skipping")
            return None, "offshore"
        crop = load_cropland_mask(bbox, wc)
        if not bool(crop.any()):
            return None, "no_cropland"                      # no cropland: skip S2
        load_bbox = _mask_extent(crop, bbox)

    items = None if s2_items is None else items_in_bbox(s2_items, load_bbox)
    scenes = load_s2_ndvi_stack(load_bbox, start, end, items=items)
    if scenes is None or scenes.sizes.get("time", 0) == 0:
        print(f"  no S2 L2A coverage for {start}..{end} at {_fmt_bbox(load_bbox)} — skipping")
        return None, "no_s2"

    cube = composite_to_anchors(scenes, start, end, align=align)
    del scenes

    if apply_cropland_mask:
        cube = cube.where(cropland_mask_on_grid(cube.isel(time=0, drop=True), mask=crop))

    ds = cube.astype("float32").to_dataset(name=NDVI_BAND)
    ds.attrs.update({"spatial_dims": ["y", "x"],
                     "window_start": str(start), "window_end": str(end),
                     "interval_days": INTERVAL_DAYS,
                     "resolution_m": round(GRID_SCALE_DEG * M_PER_DEG, 2)})
    return ds, None


# ==================================================================
# 7. Tiled province builder (memory-bounded)
# ==================================================================
def generate_tiles(bbox, tile_deg=0.1, geometry=None):
    """Regular tile grid over bbox. With `geometry` (EPSG:4326 polygon(s)),
    tiles that do not intersect it (e.g. open sea inside a coastal
    province's bounding box) are dropped."""
    w, s, e, n = bbox
    nx = max(1, int(np.ceil((e - w) / tile_deg - 1e-9)))
    ny = max(1, int(np.ceil((n - s) / tile_deg - 1e-9)))
    tiles = []
    for ix in range(nx):
        x0 = w + ix * tile_deg
        for iy in range(ny):
            y0 = s + iy * tile_deg
            tiles.append((x0, y0, min(x0 + tile_deg, e), min(y0 + tile_deg, n)))
    if geometry is not None:
        from shapely.geometry import box
        from shapely.ops import unary_union
        from shapely.prepared import prep
        shape = prep(unary_union(_as_geometry_list(geometry)))
        tiles = [tb for tb in tiles
                 if shape.intersects(box(*tb)) and not shape.touches(box(*tb))]
    return tiles


def _province_grid_coords(bbox, res=None):
    res = res or GRID_SCALE_DEG
    w, s, e, n = bbox
    xs = np.arange(w + res / 2, e, res)
    ys = np.arange(n - res / 2, s, -res)
    return ys, xs


def _mosaic_tiles(pairs, bbox, res=None):
    """
    Merge per-tile Datasets onto ONE province grid. Every data_var is
    mosaicked (2-D (y,x) products or 3-D (time,y,x) cubes). Each province
    pixel belongs to exactly one tile by its centre (half-open bins).
    """
    res = res or GRID_SCALE_DEG
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


def _cache_key(window, align, apply_cropland_mask, cache_tag=""):
    """Short hash of everything that changes a tile's output, so a cache dir
    never serves tiles built with other settings (other period end, grid,
    compositing, phenology parameters passed as cache_tag, ...)."""
    import hashlib
    parts = (window, align, apply_cropland_mask, round(GRID_SCALE_DEG, 9),
             INTERVAL_DAYS, COMPOSITE_METHOD, MAX_SCENE_CLOUD,
             tuple(SCL_CLEAR_CLASSES), str(cache_tag))
    return hashlib.md5(repr(parts).encode()).hexdigest()[:10]


def _tile_cache_path(cache_dir, tb, key):
    import os
    w, s, e, n = tb
    return os.path.join(cache_dir, f"tile_{key}_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}.npz")


def _save_tile(path, ds):
    """Tile Dataset -> .npz (vars, coords, attrs as JSON). No netCDF backend needed."""
    import json, os
    arrays = {f"var__{v}": ds[v].values for v in ds.data_vars}
    arrays.update({f"coord__{c}": ds[c].values for c in ds.coords})
    meta = {"dims": {v: list(ds[v].dims) for v in ds.data_vars},
            "attrs": {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in ds.attrs.items()}}
    arrays["meta"] = np.array(json.dumps(meta, default=str))
    tmp = path + ".part.npz"                                  # atomic write
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _save_skip(path, reason):
    """Marker for a tile with nothing to process (offshore / no cropland)."""
    import json, os
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps({"skip": reason})))
    os.replace(tmp, path)


def _load_tile(path):
    """Cached tile Dataset, or None for a skip marker."""
    import json
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        if "skip" in meta:
            return None
        coords = {k[len("coord__"):]: z[k] for k in z.files if k.startswith("coord__")}
        data = {k[len("var__"):]: (meta["dims"][k[len("var__"):]], z[k])
                for k in z.files if k.startswith("var__")}
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(meta["attrs"])
    return ds


def build_province_datacube_tiled(bbox, planting_month, year, tile_deg=0.1,
                                  per_tile_fn=None, apply_cropland_mask=True,
                                  geometry=None, tile_retries=2, cache_dir=None,
                                  allow_failed_tiles=False, tile_workers=1,
                                  window=None, align="start", cache_tag=""):
    """
    Tiled province builder.

    per_tile_fn : None  -> returns the full NDVI cube mosaic (time, y, x).
                           Fine for small AOIs; heavy for large provinces.
                  fn(ds) -> applied to each tile cube right after it is built
                           (e.g. phenology.run_phenology). Only its (small, 2-D)
                           outputs are kept and mosaicked, so peak memory is one
                           tile cube per worker. Phenology is per-pixel in time,
                           so tiling introduces no seams.
    geometry    : province polygon(s), EPSG:4326. Tiles outside it (open sea
                  inside the bbox) are not processed.
    tile_retries: extra attempts per tile after an error before it counts as failed.
    cache_dir   : if set, each finished tile's output is saved there and reused
                  on the next run, so a failed province resumes instead of
                  restarting. Offshore / no-cropland tiles are cached as skip
                  markers; tiles without S2 coverage are not cached.
                  The cache key covers the window, grid, compositing and
                  cropland settings; pass the phenology parameters as
                  `cache_tag` (e.g. repr(PHENO_CFG)) so changing them also
                  invalidates the cache.
    allow_failed_tiles : False -> raise at the end if any tile failed (the
                  province is reported failed and retried by skip_existing;
                  finished tiles stay in cache_dir). True -> mosaic what
                  succeeded and leave failed tiles NaN.
    window / align : see build_province_datacube.
    tile_workers: tiles processed in parallel threads (reads are I/O bound).
                  Peak memory scales with it: ~1-1.5 GB per worker at
                  tile_deg=0.1, so 2-3 is a safe range on standard Colab.

    STAC is searched ONCE per province (S2 + WorldCover); each tile filters
    those items locally, so a 300-tile province makes 2 searches, not 600.
    """
    import gc, os, time as _t, traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tiles = generate_tiles(bbox, tile_deg, geometry=geometry)
    n_all = len(generate_tiles(bbox, tile_deg)) if geometry is not None else len(tiles)
    print(f"  province split into {len(tiles)} tile(s) of {tile_deg} deg"
          + (f" ({n_all - len(tiles)} outside the boundary skipped)" if n_all != len(tiles) else ""))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    window = tuple(str(d) for d in (window or season_window(planting_month, year)))
    key = _cache_key(window, align, apply_cropland_mask, cache_tag)

    def cpath(tb):
        return _tile_cache_path(cache_dir, tb, key) if cache_dir else None

    todo = [tb for tb in tiles if not (cache_dir and os.path.exists(cpath(tb)))]
    s2_items = wc_items = None
    if todo:
        # one province-wide search each, over the union of the remaining tiles
        sb = (min(t[0] for t in todo), min(t[1] for t in todo),
              max(t[2] for t in todo), max(t[3] for t in todo))
        start, end = window
        s2_items = search_s2_items(sb, start, end)
        if apply_cropland_mask:
            # a province is on land, so an empty result here is transient MPC
            # trouble — let NoWorldCoverError fail the province (retried later)
            # rather than skipping every tile as "offshore"
            wc_items = worldcover_items(sb)
        print(f"  STAC: {len(s2_items)} S2 item(s)"
              + (f", {len(wc_items)} WorldCover item(s)" if apply_cropland_mask else ""))
        if not s2_items:
            # an empty province-wide search is almost certainly transient MPC
            # trouble (the retry already ran) — fail so skip_existing retries
            raise RuntimeError(f"no S2 L2A items for province bbox {_fmt_bbox(sb)} "
                               f"{start}..{end}")

    def run_tile(tb):
        """-> (tile Dataset | None, error repr | None, from_cache)"""
        p = cpath(tb)
        if p and os.path.exists(p):
            return _load_tile(p), None, True
        for attempt in range(tile_retries + 1):
            try:
                ds_t, skip = _build_tile(tb, planting_month, year, apply_cropland_mask,
                                         s2_items=s2_items, wc_items=wc_items,
                                         window=window, align=align)
                if p and skip in _PERMANENT_SKIPS and wc_items:
                    _save_skip(p, skip)     # decided from a successful WorldCover search
                if ds_t is not None:
                    cube_attrs = dict(ds_t.attrs)
                    if per_tile_fn is not None:
                        ds_t = per_tile_fn(ds_t)
                        ds_t.attrs.update({k: v for k, v in cube_attrs.items()
                                           if k not in ds_t.attrs})
                    if p:
                        _save_tile(p, ds_t)
                return ds_t, None, False
            except Exception as e:
                if attempt < tile_retries:
                    print(f"  tile {_fmt_bbox(tb)}: {e.__class__.__name__} "
                          f"— retry {attempt + 1}/{tile_retries}")
                    _t.sleep(5 * 2 ** attempt)
                else:
                    print(f"  tile {_fmt_bbox(tb)} FAILED: {e!r}")
                    traceback.print_exc()
                    return None, repr(e), False
            finally:
                gc.collect()

    pairs, attrs, failed, n_cached = [], {}, [], 0

    def collect(tb, res):
        nonlocal attrs, n_cached
        ds_t, err, cached = res
        n_cached += cached
        if err is not None:
            failed.append((tb, err))
        elif ds_t is not None:
            attrs = dict(ds_t.attrs) or attrs
            pairs.append((ds_t, tb))

    bar = tqdm(total=len(tiles), desc="tiles", unit="tile")
    if tile_workers and tile_workers > 1:
        with ThreadPoolExecutor(max_workers=tile_workers) as pool:
            futs = {pool.submit(run_tile, tb): tb for tb in tiles}
            for f in as_completed(futs):
                collect(futs[f], f.result())
                bar.update(1)
    else:
        for tb in tiles:
            collect(tb, run_tile(tb))
            bar.update(1)
    bar.close()

    if n_cached:
        print(f"  {n_cached} tile(s) reused from cache")
    if failed and not allow_failed_tiles:
        raise RuntimeError(
            f"{len(failed)}/{len(tiles)} tile(s) failed (first: {failed[0][1]})"
            + ("; finished tiles are cached and will be reused on rerun" if cache_dir else ""))
    ds = _mosaic_tiles(pairs, bbox)
    if ds is not None:
        for k, v in attrs.items():
            ds.attrs.setdefault(k, v)
        if failed:
            ds.attrs["failed_tiles"] = len(failed)
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
