"""
pipeline.py
==================================================================
The two run modes, on top of data_processing (S2 NDVI cubes) and
phenology (transition dates -> growth stages).

PERIODIC  — large areas: nationwide, regional or provincial
    Run on a schedule (monthly, or semimonthly = 1st-15th / 16th-end of
    month). Each run processes every province in the selection with the
    season method (season window from the province's planting month), using
    data up to `as_of`, and writes ONE stage map per province for each
    requested period (by default the last completed period).
        run_periodic_unit(...) -> phenology Dataset
        stage_maps(pheno, periods, method) -> [(Period, stage DataArray)]

RECENT    — small areas: municipal or barangay ("near real time")
    On demand. A rolling window that ends today (lookback_days long, not
    tied to a planting month), composites anchored on the newest data, and
    the most recent crop cycle (PEAK_SELECT='last'). Output: the growth stage
    on `as_of` plus how many days old the newest clear observation is.
        run_recent(...) -> (recent Dataset, phenology Dataset)
        stage_area_summary(stage) -> hectares per stage

Admin-unit selection for both: select_units(gdf, level, names, ...).
==================================================================
"""

import re
from collections import namedtuple

import numpy as np
import pandas as pd

from . import data_processing as dp
from . import phenology as ph

CADENCES = ("monthly", "semimonthly")
Period = namedtuple("Period", "start end label")   # [start, end), label for file names

# phenology overrides for the recent (rolling-window) mode
RECENT_LOOKBACK_DAYS = 240
RECENT_PHENO = dict(PEAK_SELECT="last")


# ==================================================================
# Periods
# ==================================================================
def period_of(date, cadence="monthly"):
    """Period containing `date`. monthly -> label 'YYYYMM';
    semimonthly -> 'YYYYMMH1' (days 1-15) or 'YYYYMMH2' (16-end)."""
    if cadence not in CADENCES:
        raise ValueError(f"cadence must be one of {CADENCES}")
    d = pd.Timestamp(date).normalize()
    m0 = d.replace(day=1)
    m1 = m0 + pd.offsets.MonthBegin(1)
    if cadence == "monthly":
        return Period(m0, m1, m0.strftime("%Y%m"))
    mid = m0 + pd.Timedelta(days=15)
    if d < mid:
        return Period(m0, mid, m0.strftime("%Y%m") + "H1")
    return Period(mid, m1, m0.strftime("%Y%m") + "H2")


def last_complete_period(as_of, cadence="monthly"):
    """Newest period that has fully ended by `as_of` (data through as_of).
    Run on the 1st of a month (monthly) -> the previous month."""
    d = pd.Timestamp(as_of).normalize()
    cur = period_of(d, cadence)
    if d + pd.Timedelta(days=1) >= cur.end:            # as_of is its last day
        return cur
    return period_of(cur.start - pd.Timedelta(days=1), cadence)


def periods_between(start, end, cadence="monthly"):
    """All periods overlapping [start, end] (dates inclusive)."""
    out, p = [], period_of(start, cadence)
    end = pd.Timestamp(end).normalize()
    while p.start <= end:
        out.append(p)
        p = period_of(p.end, cadence)
    return out


# ==================================================================
# Cropping seasons
# ==================================================================
# Philippine rice seasons and the province-table column holding each one's
# planting month:
#   dry : Semester_1, planted ~Oct-Dec (some provinces Jan-Mar), harvested the next year
#   wet : Semester_2, planted ~May-Jun, harvested the same year
# A season is named by its HARVEST year: dry2026 = planted Oct-Dec 2025 (or Jan-Mar 2026),
# wet2026 = planted May-Jun 2026.
SEASONS = ("dry", "wet")
SEASON_COLUMNS = {"dry": "Semester_1", "wet": "Semester_2"}
SEASON_BY_COLUMN = {v.upper(): k for k, v in SEASON_COLUMNS.items()}
SEASON_FALLBACK_MONTH = {"dry": 12, "wet": 6}   # when a province has no planting month
SEASON_TYPICAL_MONTHS = {"dry": {9, 10, 11, 12, 1, 2, 3}, "wet": {4, 5, 6, 7, 8}}
DRY_PREV_YEAR_FROM = 7                           # dry season: months Jul-Dec are the year before

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def season_from_column(col):
    """'Semester_1' -> 'dry', 'Semester_2' -> 'wet', anything else -> None."""
    return SEASON_BY_COLUMN.get(str(col or "").strip().upper())


def parse_month(value):
    """Planting month from a table cell: 5, 5.0, '5', 'May', 'MAY', 'June', '5-6',
    'May-June' (first month of a range). None when empty / unreadable / 0."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        m = int(round(float(value)))
        return m if 1 <= m <= 12 else None
    text = str(value).strip().lower()
    num = re.match(r"^(\d{1,2})(?:\.0+)?(?:\s*[-/–]\s*\d{1,2})?$", text)
    if num:
        m = int(num.group(1))
        return m if 1 <= m <= 12 else None
    word = re.match(r"^([a-z]{3,})", text)
    if word:
        return _MONTHS.get(word.group(1)[:3])
    return None


def planting_year(month, season, season_year):
    """Calendar year of a planting month in a season named by its harvest year:
    wet -> season_year; dry -> season_year - 1 for Jul-Dec, season_year for Jan-Jun."""
    if season == "dry" and month >= DRY_PREV_YEAR_FROM:
        return season_year - 1
    return season_year


def default_season_year(season, today=None):
    """The season in progress / most recently started at `today`."""
    t = pd.Timestamp(today or pd.Timestamp.today())
    if season == "dry":
        return t.year + 1 if t.month >= DRY_PREV_YEAR_FROM else t.year
    return t.year


def season_window_for_months(months, season, season_year):
    """One window covering several planting months of a season (e.g. a province
    whose municipalities plant in Oct, Nov and Dec): from 1 month before the
    earliest planting to 6 months after the latest. Returns (start, end, earliest_month)."""
    wins = sorted((dp.season_window(m, planting_year(m, season, season_year)), m) for m in months)
    return wins[0][0][0], max(w[1] for w, _ in wins), wins[0][1]


def season_label(season, season_year):
    return f"{season}{season_year}"


def season_prefix(season, season_year, root="products/growth_stage"):
    return f"{root}/{season_year}/{season}"


# ==================================================================
# Admin units
# ==================================================================
LEVELS = ("national", "regional", "provincial")


def select_units(gdf, level="national", names=None, name_col="Pro_Name",
                 region_col=None, exclude=()):
    """
    Rows of `gdf` (one per province) to process.
      national   : every province (minus names containing an `exclude` entry)
      regional   : provinces whose `region_col` is in `names`
      provincial : provinces whose `name_col` is in `names`
    Names are matched case-insensitively. Processing and outputs stay per
    province (memory / planting month); the level only picks which ones.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    norm = lambda s: s.astype(str).str.strip().str.upper()
    sel = gdf.copy()
    if exclude:                                        # substring match, e.g. 'PALAWAN'
        import re
        pat = "|".join(re.escape(e.strip().upper()) for e in exclude)
        sel = sel[~norm(sel[name_col]).str.contains(pat, na=False)]
    if level == "national":
        return sel
    if not names:
        raise ValueError(f"level={level!r} needs a list of names")
    want = [n.strip().upper() for n in names]
    col = name_col if level == "provincial" else region_col
    if level == "regional" and not col:
        raise ValueError("level='regional' needs region_col")
    picked = sel[norm(sel[col]).isin(want)]
    missing = sorted(set(want) - set(norm(picked[col])))
    if missing:
        raise ValueError(f"not found in {col!r}: {missing}")
    return picked


def _geometry_and_bbox(area):
    """GeoDataFrame / GeoSeries / shapely geometry / (w, s, e, n) bbox ->
    (geometry for tiling + clipping, bbox)."""
    if isinstance(area, (tuple, list)) and len(area) == 4 and \
            all(isinstance(v, (int, float, np.floating)) for v in area):
        from shapely.geometry import box
        return box(*area), tuple(float(v) for v in area)
    if hasattr(area, "total_bounds"):
        return area, tuple(float(v) for v in area.total_bounds)
    return area, tuple(float(v) for v in area.bounds)


def _cfg_tag(cfg):
    return repr(sorted((k, repr(v)) for k, v in (cfg or {}).items()))


# ==================================================================
# PERIODIC mode (national / regional / provincial)
# ==================================================================
def periodic_window(planting_month, year, as_of, window=None):
    """Season window of (planting_month, year) — or the given (start, end) —
    clipped at as_of; None if the season has not started by as_of."""
    s, e = window or dp.season_window(planting_month, year)
    as_of = pd.Timestamp(as_of).normalize()
    if as_of < pd.Timestamp(s):
        return None
    return s, str(min(pd.Timestamp(e), as_of).date())


def run_periodic_unit(area, planting_month, year, as_of, tile_deg=0.25,
                      pheno_cfg=None, cache_dir=None, tile_workers=1,
                      date_median_radius=1, window=None):
    """
    Season phenology for one admin unit (province) with data up to as_of.
    Returns the phenology Dataset clipped to the unit, or None (season not
    started / no coverage). attrs['season_start'/'season_end'] keep the full
    season window for listing its periods.
    window : (start, end) season window to use instead of season_window(planting_month,
             year), e.g. one covering several planting months (season_window_for_months).
    """
    win = periodic_window(planting_month, year, as_of, window)
    if win is None:
        return None
    geom, bbox = _geometry_and_bbox(area)
    cfg = dict(pheno_cfg or {})
    pheno = dp.build_province_datacube_tiled(
        bbox, planting_month, year, tile_deg=tile_deg, geometry=geom,
        cache_dir=cache_dir, tile_workers=tile_workers, window=win,
        cache_tag=_cfg_tag(cfg),
        per_tile_fn=lambda ds: ph.run_phenology(ds, **cfg))
    if pheno is None:
        return None
    pheno = ph.smooth_dates(pheno, date_median_radius)
    pheno = dp.clip_to_geometry(pheno, geom)
    s, e = window or dp.season_window(planting_month, year)
    pheno.attrs.update(season_start=s, season_end=e, as_of=str(pd.Timestamp(as_of).date()))
    return pheno


def season_periods(planting_month, year, cadence="monthly", until=None):
    """Periods of the season window, optionally only those that ended by `until`."""
    s, e = dp.season_window(planting_month, year)
    ps = periods_between(s, e, cadence)
    if until is not None:
        u = pd.Timestamp(until).normalize() + pd.Timedelta(days=1)
        ps = [p for p in ps if p.end <= u]
    return ps


def stage_maps(pheno, periods, method="dominant"):
    """[(Period, int16 stage DataArray)] — one map per period."""
    return [(p, ph.classify_stage_period(pheno, p.start, p.end, method)) for p in periods]


# ==================================================================
# RECENT mode (municipal / barangay, near real time)
# ==================================================================
def run_recent(area, as_of=None, lookback_days=RECENT_LOOKBACK_DAYS, tile_deg=0.1,
               pheno_cfg=None, cache_dir=None, tile_workers=1, date_median_radius=1):
    """
    Most recent growth stage of the crop in `area` (geometry or bbox).

    Window: [as_of - lookback_days, as_of] (as_of default: today, UTC).
    Composites are anchored on as_of so the newest scenes are used, and the
    most recent cycle is followed (PEAK_SELECT='last'), so a young crop is not
    mistaken for the previous harvested one.

    Returns (recent, pheno): recent = phenology.recent_stage() Dataset
    (growth_stage, data_age_days, qc) clipped to the area; pheno = the full
    phenology Dataset (transition dates etc.). (None, None) if no coverage.
    """
    as_of = pd.Timestamp(as_of or pd.Timestamp.now("UTC").tz_localize(None)).normalize()
    start = as_of - pd.Timedelta(days=int(lookback_days))
    win = (str(start.date()), str(as_of.date()))
    cfg = dict(pheno_cfg or {})
    cfg.update(RECENT_PHENO)
    # whole window (a season-style range would cut off the newest cycle)
    cfg["PEAK_SEARCH_DAYS"] = (30, int(lookback_days))
    geom, bbox = _geometry_and_bbox(area)
    pheno = dp.build_province_datacube_tiled(
        bbox, None, as_of.year, tile_deg=tile_deg, geometry=geom,
        cache_dir=cache_dir, tile_workers=tile_workers, window=win, align="end",
        cache_tag=_cfg_tag(cfg),
        per_tile_fn=lambda ds: ph.run_phenology(ds, **cfg))
    if pheno is None:
        return None, None
    pheno = ph.smooth_dates(pheno, date_median_radius)
    pheno = dp.clip_to_geometry(pheno, geom)
    return ph.recent_stage(pheno, as_of), pheno


def stage_area_summary(stage):
    """Pixels, hectares and share of mapped cropland per stage class for a
    stage DataArray (int16, -1 = nodata). Pixel area follows latitude."""
    a = np.asarray(stage.values)
    res = dp.GRID_SCALE_DEG
    lat = np.asarray(stage["y"].values, dtype=float)
    px_ha = (res * dp.M_PER_DEG) ** 2 * np.cos(np.radians(lat)) / 1e4   # per row
    rows = []
    for code, name in ph.STAGE_CLASSES.items():
        m = a == code
        rows.append({"stage": code, "name": name, "pixels": int(m.sum()),
                     "hectares": float((m * px_ha[:, None]).sum())})
    df = pd.DataFrame(rows)
    total = df["hectares"].sum()
    df["share_%"] = (100 * df["hectares"] / total).round(1) if total else 0.0
    df["hectares"] = df["hectares"].round(1)
    return df
