"""
runner.py
==================================================================
One object that runs the growth-stage products end to end and saves them to
an output store (GCS, Google Drive, both, or a local folder). Used by the
notebook directly and by the agent (agent.py) as its tools.

    cfg = RunConfig(vector_path=..., output_target='gdrive', ...)
    runner = Runner(cfg, gcs_client=client)          # client only for GCS output
    runner.run_periodic('regional', ['REGION VII'])  # national / regional / provincial
    runner.run_recent(['SCIENCE CITY OF MUÑOZ'])     # municipal / barangay, current stage
==================================================================
"""

import gc
import os
import re
import shutil
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from . import data_processing as dp
from . import phenology as ph
from . import pipeline as pl
from . import output_store as storage
from .mosaic import GridWriter, mosaic_cogs

DEFAULT_PHENO_CFG = dict(
    THRESHOLD_FRAC=0.15, BASE_MODE="separate", SG_WINDOW=5, SG_POLYORDER=2,
    MIN_AMPLITUDE=0.20, MIN_PEAK_NDVI=0.50, MAX_BASE_NDVI=0.45,
)


@dataclass
class RunConfig:
    # data / grid
    data_source: str = "hls"               # 'hls' | 's2'
    resolution_m: float = 30
    tile_deg: float = None                 # None -> 0.25 deg at >= 20 m, else 0.1
    tile_workers: int = 2
    interval_days: int = 10
    composite_method: str = "max"
    date_median_radius: int = 1
    stage_majority_radius: int = 0
    pheno_cfg: dict = field(default_factory=lambda: dict(DEFAULT_PHENO_CFG))
    tile_cache_dir: str = None

    # periodic (national / regional / provincial)
    vector_path: str = None
    prov_col: str = "Pro_Name"
    region_col: str = None
    # season: which planting-month column, and which year (see pipeline.SEASONS)
    #   plant_mo_col  'Semester_1' = dry season (planted ~Oct-Dec), 'Semester_2' = wet (~May-Jun)
    #   season        'dry' | 'wet'; None -> from plant_mo_col (and plant_mo_col None -> from season)
    #   season_year   HARVEST year naming the season: dry2026 = planted Oct-Dec 2025 (or Jan-Mar
    #                 2026); wet2026 = planted May-Jun 2026. None -> from `year`, else the season
    #                 in progress today
    #   year          legacy: the PLANTING year of the season's main months (2025 -> dry2026 / wet2025)
    plant_mo_col: str = None
    vector_layer: str = None               # layer of a multi-layer GeoPackage; None -> the
                                           #   layer that has the province + planting-month columns
    season: str = None
    season_year: int = None
    year: int = None
    exclude: tuple = ()                    # provinces to skip (substring match), e.g. ("PALAWAN",)
    planting_month_fallback: int = None    # None -> 12 (dry) / 6 (wet)
    cadence: str = "monthly"               # 'monthly' | 'semimonthly'
    periods: str = "last_complete"         # 'last_complete' | 'current' | 'season'
    stage_method: str = "dominant"         # 'dominant' | 'midpoint'
    save_phenology_bands: bool = False
    mosaic_after: bool = True              # after a periodic run, mosaic the provinces into one COG
                                           #   per period (national / regional / multi-province runs)
    cog_label: str = None                  # None -> '<season><season_year>', e.g. 'wet2026'
    output_prefix: str = None              # None -> 'products/growth_stage/<season_year>/<season>'

    # recent (municipal / barangay)
    aoi_path: str = None
    aoi_name_col: str = "Mun_Name"
    lookback_days: int = 240
    recent_prefix: str = "products/growth_stage/recent"

    # outputs
    output_target: str = "gcs"             # 'gcs' | 'gdrive' | 'both' | 'local'
    gcs_bucket: str = None
    drive_root: str = storage.DEFAULT_DRIVE_ROOT
    local_root: str = "outputs"

    # ---- season resolution
    def season_name(self):
        by_col = pl.season_from_column(self.plant_mo_col)
        if self.season:
            if self.season not in pl.SEASONS:
                raise ValueError(f"season must be one of {list(pl.SEASONS)}")
            if by_col and by_col != self.season:
                raise ValueError(f"plant_mo_col {self.plant_mo_col!r} is the {by_col} season's column, "
                                 f"but season={self.season!r}")
            return self.season
        return by_col or "dry"

    def plant_col(self):
        return self.plant_mo_col or pl.SEASON_COLUMNS[self.season_name()]

    def season_year_value(self):
        if self.season_year:
            return int(self.season_year)
        if self.year:                                   # legacy: planting year of the main months
            return int(self.year) + (1 if self.season_name() == "dry" else 0)
        return pl.default_season_year(self.season_name())

    def planting_year(self, month):
        return pl.planting_year(month, self.season_name(), self.season_year_value())

    def fallback_month(self):
        return int(self.planting_month_fallback or pl.SEASON_FALLBACK_MONTH[self.season_name()])

    def label(self):
        return self.cog_label or pl.season_label(self.season_name(), self.season_year_value())

    def prefix(self):
        return self.output_prefix or pl.season_prefix(self.season_name(), self.season_year_value())

    def ref_year(self):
        """Year whose Jan 1 the QA date bands count from (the planting year of the main months)."""
        return self.planting_year(pl.SEASON_FALLBACK_MONTH[self.season_name()])

    def season_summary(self):
        f = self.fallback_month()
        s, e = dp.season_window(f, self.planting_year(f))
        return {"season": self.season_name(), "season_year": self.season_year_value(),
                "label": self.label(), "planting_month_column": self.plant_col(),
                "fallback_planting_month": f, "window_for_fallback_month": f"{s}..{e}",
                "output_prefix": self.prefix()}

    def effective_tile_deg(self):
        return self.tile_deg or (0.25 if self.resolution_m >= 20 else 0.1)

    def apply(self):
        """Push the processing settings into the data_processing module."""
        dp.set_data_source(self.data_source)
        dp.set_resolution(self.resolution_m)
        dp.INTERVAL_DAYS = int(self.interval_days)
        dp.COMPOSITE_METHOD = self.composite_method

    def public(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _norm_col(name):
    return re.sub(r"[\s_\-]+", "_", str(name).strip()).lower()


def _match_columns(columns, wanted):
    """{wanted: actual} for each wanted column found exactly or ignoring case / spaces."""
    by_norm = {}
    for col in columns:
        by_norm.setdefault(_norm_col(col), col)
    out = {}
    for w in wanted:
        if w in columns:
            out[w] = w
        elif _norm_col(w) in by_norm:
            out[w] = by_norm[_norm_col(w)]
    return out


def _list_layers(path):
    try:
        import pyogrio
        return [str(l[0]) for l in pyogrio.list_layers(path)]
    except Exception:
        try:
            import fiona
            return list(fiona.listlayers(path))
        except Exception:
            return [None]


def _layer_columns(path, layer):
    import geopandas as gpd
    try:
        return [c for c in gpd.read_file(path, layer=layer, rows=1).columns if c != "geometry"]
    except TypeError:                                    # very old geopandas: no `rows`
        return [c for c in gpd.read_file(path, layer=layer).columns if c != "geometry"]


def read_vector(path, need, optional=(), layer=None, log=print):
    """Read the layer of `path` that has every column in `need` (case / space
    insensitive), renaming matched columns to the requested names. Raises a
    ValueError listing every layer's columns when none has them."""
    import geopandas as gpd
    layers = [layer] if layer else _list_layers(path)
    report = {}
    for lyr in layers:
        cols = _layer_columns(path, lyr)
        found = _match_columns(cols, list(need) + [o for o in optional if o])
        report[lyr] = cols
        if all(n in found for n in need):
            g = gpd.read_file(path, layer=lyr).to_crs(4326)
            rename = {a: w for w, a in found.items() if a != w}
            if rename:
                log(f"  columns matched: {rename}")
                g = g.rename(columns=rename)
            if len(layers) > 1:
                log(f"  {os.path.basename(path)}: using layer {lyr!r}")
            return g
    try:
        modified = pd.Timestamp(os.path.getmtime(path), unit="s").strftime("%Y-%m-%d %H:%M UTC")
    except OSError:
        modified = "unknown"
    lines = "\n".join(f"  layer {k!r}: {v}" for k, v in report.items())
    missing = sorted({n for n in need for cols in [report[next(iter(report))]]
                      if n not in _match_columns(cols, [n])})
    raise ValueError(
        f"{path} (last modified {modified}) has no layer with the column(s) {list(need)}; "
        f"missing in the first layer: {missing}.\n{lines}\n"
        "If you added the column recently, Colab may still be reading the old copy of the file "
        "from Google Drive: run `drive.mount('/content/drive', force_remount=True)` (or restart the "
        "runtime) and check the path; for a multi-layer GeoPackage set vector_layer.")


def _key(name):
    return str(name).strip().upper()


def _fname(name):
    return str(name).strip().replace(" ", "_").replace("/", "-")


class Runner:
    def __init__(self, cfg, gcs_client=None, log=print):
        self.cfg, self.gcs_client, self.log = cfg, gcs_client, log
        self._store, self._prov_gdf, self._aoi_gdf = None, None, None
        self.last_periodic, self.last_recent = None, None
        cfg.apply()

    # ------------------------------------------------------------ setup
    @property
    def store(self):
        if self._store is None:
            c = self.cfg
            self._store = storage.make_store(c.output_target, gcs_client=self.gcs_client,
                                             gcs_bucket=c.gcs_bucket, drive_root=c.drive_root,
                                             local_root=c.local_root)
        return self._store

    def update(self, **changes):
        """Change settings (validated); rebuilds the store / data settings as needed."""
        valid = {f.name for f in fields(RunConfig)}
        bad = sorted(set(changes) - valid)
        if bad:
            raise ValueError(f"unknown setting(s) {bad}")
        checks = {"data_source": ("hls", "s2"), "cadence": pl.CADENCES, "season": pl.SEASONS,
                  "periods": ("last_complete", "current", "season"),
                  "stage_method": ("dominant", "midpoint"),
                  "output_target": storage.TARGETS, "composite_method": ("max", "median")}
        for k, allowed in checks.items():
            if k in changes and changes[k] not in allowed:
                raise ValueError(f"{k} must be one of {list(allowed)}")
        if "pheno_cfg" in changes:
            unknown = sorted(set(changes["pheno_cfg"]) - set(ph.DEFAULTS))
            if unknown:
                raise ValueError(f"unknown phenology parameter(s) {unknown}")
            merged = dict(self.cfg.pheno_cfg); merged.update(changes.pop("pheno_cfg"))
            self.cfg.pheno_cfg = merged
        if "season" in changes and "plant_mo_col" not in changes and \
                pl.season_from_column(self.cfg.plant_mo_col) not in (None, changes["season"]):
            changes["plant_mo_col"] = pl.SEASON_COLUMNS[changes["season"]]   # follow the season
        old = self.cfg.public()
        for k, v in changes.items():
            setattr(self.cfg, k, v)
        try:
            self.cfg.season_name()
        except ValueError:
            for k, v in old.items():
                setattr(self.cfg, k, v)
            raise
        if {"output_target", "gcs_bucket", "drive_root", "local_root"} & set(changes):
            self._store = None
            self.store                                       # fail now, not mid-run
        if {"vector_path", "vector_layer", "prov_col", "plant_mo_col", "season",
                "region_col"} & set(changes):
            self._prov_gdf = None
        if {"aoi_path", "aoi_name_col"} & set(changes):
            self._aoi_gdf = None
        self.cfg.apply()
        return self.cfg.public()

    def provinces_gdf(self):
        """Province table (one or more rows per province, e.g. one per municipality).
        Picks the GeoPackage layer that has the needed columns and matches column
        names ignoring case / spaces ('semester 2' -> 'Semester_2')."""
        if self._prov_gdf is None:
            c = self.cfg
            if not c.vector_path:
                raise ValueError("vector_path (province boundaries) is not set")
            need = [c.prov_col, c.plant_col()]
            g = read_vector(c.vector_path, need, optional=[c.region_col] if c.region_col else [],
                            layer=c.vector_layer, log=self.log)
            g[c.prov_col] = g[c.prov_col].map(_key)
            self._prov_gdf = g
        return self._prov_gdf

    def aoi_gdf(self):
        if self._aoi_gdf is None:
            import geopandas as gpd
            if not self.cfg.aoi_path:
                raise ValueError("aoi_path (municipal / barangay boundaries) is not set")
            g = gpd.read_file(self.cfg.aoi_path).to_crs(4326)
            g[self.cfg.aoi_name_col] = g[self.cfg.aoi_name_col].map(_key)
            self._aoi_gdf = g
        return self._aoi_gdf

    # ------------------------------------------------------------ discovery
    def list_areas(self, kind="province", region=None, contains=None):
        """Names available: kind 'province' | 'region' | 'aoi' (municipal /
        barangay file). Optional region filter (provinces) / substring."""
        c = self.cfg
        if kind == "aoi":
            names = self.aoi_gdf()[c.aoi_name_col]
        else:
            g = self.provinces_gdf()
            if kind == "region":
                if not c.region_col or c.region_col not in g:
                    raise ValueError(f"region column {c.region_col!r} not in the province file; "
                                     f"columns: {list(g.columns)}")
                names = g[c.region_col].astype(str).str.strip()
            else:
                if region:
                    g = g[g[c.region_col].astype(str).map(_key) == _key(region)]
                names = g[c.prov_col]
        names = sorted(set(names.dropna().astype(str)))
        if contains:
            names = [n for n in names if _key(contains) in _key(n)]
        return names

    def select(self, level, names=None):
        c = self.cfg
        return pl.select_units(self.provinces_gdf(), level, names, name_col=c.prov_col,
                               region_col=c.region_col, exclude=c.exclude)

    # ------------------------------------------------------------ periodic
    def _planting(self, units, prov):
        """Planting months of a province from ALL its rows (a municipal-level table has
        one row per municipality) -> {'pm': earliest month, 'months': {month: rows},
        'window': (start, end) covering them, 'note': str | None}.
        Months on fewer than 10 % of the rows are treated as outliers; empty /
        unreadable cells are ignored; with no usable value the season's fallback is used."""
        c = self.cfg
        col = c.plant_col()
        if col not in units:
            raise ValueError(f"planting-month column {col!r} not in the province file; "
                             f"columns: {[x for x in units.columns if x != 'geometry']}")
        raw = units.loc[units[c.prov_col] == prov, col]
        parsed = [pl.parse_month(v) for v in raw]
        good = [m for m in parsed if m]
        notes = []
        if not good:
            months, counts = [c.fallback_month()], {}
            notes.append(f"no usable value in {col} ({raw.iloc[0]!r}): fallback month {months[0]}")
        else:
            counts = dict(sorted(Counter(good).items()))
            months = [m for m, n in counts.items() if n >= max(1, 0.1 * len(good))]
            rare = sorted(set(counts) - set(months))
            if rare:
                notes.append(f"months {rare} on <10% of rows ignored")
            if len(good) < len(parsed):
                notes.append(f"{len(parsed) - len(good)} row(s) without a usable month")
            odd = [m for m in months if m not in pl.SEASON_TYPICAL_MONTHS[c.season_name()]]
            if odd:
                notes.append(f"month(s) {odd} unusual for the {c.season_name()} season")
        s, e, pm = pl.season_window_for_months(months, c.season_name(), c.season_year_value())
        return {"pm": pm, "months": counts, "window": (s, e), "note": "; ".join(notes) or None}

    def _pm_info(self, units, prov):
        info = self._planting(units, prov)
        return info["pm"], info["note"]

    def _pm(self, units, prov):
        return self._planting(units, prov)["pm"]

    def window_for(self, pm):
        return dp.season_window(pm, self.cfg.planting_year(pm))

    def periods_for(self, window, as_of):
        """Periods to map for a season window (start, end) — or a planting month."""
        c = self.cfg
        s, e = window if isinstance(window, tuple) else self.window_for(window)
        season = pl.periods_between(s, e, c.cadence)
        if c.periods == "season":
            u = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=1)
            ps = [p for p in season if p.end <= u]
        elif c.periods == "current":
            ps = [pl.period_of(as_of, c.cadence)]
        else:
            ps = [pl.last_complete_period(as_of, c.cadence)]
        return [p for p in ps if p in season]

    def _stage_rel(self, prov, period):
        c = self.cfg
        return f"{c.prefix()}/{c.label()}_{_fname(prov)}_{period.label}.tiff"

    def _summary_rel(self, prov, period):
        c = self.cfg
        return f"{c.prefix()}/summary/{c.label()}_{_fname(prov)}_{period.label}.csv"

    def _qa_rel(self, prov):
        c = self.cfg
        return f"{c.prefix()}/qa/{c.label()}_{_fname(prov)}_phenology_dates.tiff"

    @staticmethod
    def _as_of(as_of):
        return pd.Timestamp(as_of or pd.Timestamp.today()).normalize()

    def plan_periodic(self, level="national", names=None, as_of=None):
        """What a periodic run would do, without downloading anything."""
        as_of = self._as_of(as_of)
        units = self.select(level, names)
        tile = self.cfg.effective_tile_deg()
        rows, n_tiles = [], 0
        for prov in sorted(units[self.cfg.prov_col]):
            sub = units.loc[units[self.cfg.prov_col] == prov]
            info = self._planting(units, prov)
            pm, note = info["pm"], info["note"]
            ps = self.periods_for(info["window"], as_of)
            nt = len(dp.generate_tiles(tuple(sub.total_bounds), tile, geometry=sub))
            n_tiles += nt if ps else 0
            s, e = info["window"]
            rows.append({"province": prov, "planting_month": pm, "window": f"{s}..{e}", "tiles": nt,
                         **({"planting_months": info["months"]} if len(info["months"]) > 1 else {}),
                         **({"note": note} if note else {}),
                         "periods": [p.label for p in ps]})
        return {"level": level, "names": names or [], "as_of": str(as_of.date()),
                "provinces": len(rows), "provinces_in_season": sum(bool(r["periods"]) for r in rows),
                "tiles_to_process": n_tiles, "tile_deg": tile,
                "data_source": self.cfg.data_source, "resolution_m": self.cfg.resolution_m,
                "cadence": self.cfg.cadence, "periods_mode": self.cfg.periods,
                "season": self.cfg.label(), "planting_month_column": self.cfg.plant_col(),
                "notes": sum("note" in r for r in rows),
                "output": f"{self.store.describe()}/{self.cfg.prefix()}",
                "detail": rows}

    def process_province(self, units, prov, as_of):
        c = self.cfg
        sub = units.loc[units[c.prov_col] == prov]
        cache = os.path.join(c.tile_cache_dir, _fname(prov)) if c.tile_cache_dir else None
        info = self._planting(units, prov)
        pm = info["pm"]
        pheno = pl.run_periodic_unit(sub, pm, c.planting_year(pm), as_of, window=info["window"],
                                     tile_deg=c.effective_tile_deg(), pheno_cfg=c.pheno_cfg,
                                     cache_dir=cache, tile_workers=c.tile_workers,
                                     date_median_radius=c.date_median_radius)
        if pheno is not None:
            pheno.attrs["province"] = prov
        return pheno

    def save_periodic(self, pheno, prov, periods):
        """Stage COG + hectares CSV per period (and the optional QA file)."""
        c, out = self.cfg, []
        tmp = tempfile.mkdtemp(prefix="cropgrowth_")
        try:
            for p, st in pl.stage_maps(pheno, periods, c.stage_method):
                if c.stage_majority_radius:
                    st = st.copy(data=ph.majority_filter(st.values, c.stage_majority_radius))
                local = os.path.join(tmp, os.path.basename(self._stage_rel(prov, p)))
                ph.write_stage_cog(st, local)
                uri = self.store.put(local, self._stage_rel(prov, p))
                summ = pl.stage_area_summary(st)
                csv = os.path.join(tmp, os.path.basename(self._summary_rel(prov, p)))
                summ.to_csv(csv, index=False)
                self.store.put(csv, self._summary_rel(prov, p))
                out.append({"period": p.label, "uri": uri,
                            "hectares": {r["name"]: r["hectares"] for _, r in summ.iterrows()
                                         if r["hectares"] > 0}})
            if c.save_phenology_bands:
                local = os.path.join(tmp, os.path.basename(self._qa_rel(prov)))
                ph.write_cog(pheno, local, ref_year=c.ref_year())
                self.store.put(local, self._qa_rel(prov))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return out

    def stream_province(self, units, prov, as_of, periods):
        """
        Province products without holding the province in memory (needed for large ones such
        as Palawan). Each finished tile is smoothed, clipped to the province outline, classified
        into a stage map per period and written straight into a disk-backed raster on the
        global grid (mosaic.GridWriter); the COGs are built from those at the end.
        Peak memory: one tile (per worker). Returns the same list as save_periodic, or None
        when no tile had data.
        """
        c = self.cfg
        sub = units.loc[units[c.prov_col] == prov]
        info = self._planting(units, prov)
        pm = info["pm"]
        win = pl.periodic_window(pm, c.planting_year(pm), as_of, info["window"])
        if win is None:
            return None
        bbox = tuple(float(v) for v in sub.total_bounds)
        res = dp.GRID_SCALE_DEG
        tags = {"province": prov, "season": c.label(), "window": f"{win[0]}..{win[1]}",
                "classes": "; ".join(f"{k}={v}" for k, v in ph.STAGE_CLASSES.items()),
                "method": c.stage_method, "data_source": c.data_source}
        writers = {p.label: GridWriter(bbox, res, 1, "int16", ph.STAGE_NODATA,
                                       band_names=[f"growth_stage {p.label}"],
                                       tags={**tags, "period": p.label,
                                             "period_dates": f"{p.start.date()}..{p.end.date()}"})
                   for p in periods}
        qa = (GridWriter(bbox, res, len(ph.QA_BANDS), "int16", ph.QA_NODATA, band_names=ph.QA_BANDS,
                         tags={**tags, **ph.qa_tags(c.ref_year(), f"{win[0]}..{win[1]}")})
              if c.save_phenology_bands else None)
        all_writers = list(writers.values()) + ([qa] if qa else [])
        pcfg = dict(c.pheno_cfg)

        def sink(t, tb):
            w, s_, e, n = tb                         # each pixel belongs to exactly one tile
            t = t.isel(x=np.where((t.x.values >= w) & (t.x.values < e))[0],
                       y=np.where((t.y.values > s_) & (t.y.values <= n))[0])
            if t.sizes["x"] == 0 or t.sizes["y"] == 0:
                return
            t = ph.smooth_dates(t, c.date_median_radius)
            t = dp.clip_to_geometry(t, sub, drop=False)
            ys, xs = t["y"].values, t["x"].values
            for p in periods:
                st = ph.classify_stage_period(t, p.start, p.end, c.stage_method)
                if c.stage_majority_radius:
                    st = st.copy(data=ph.majority_filter(st.values, c.stage_majority_radius))
                writers[p.label].write(st.values, ys, xs)
            if qa is not None:
                qa.write(ph.encode_qa_bands(t, c.ref_year()), ys, xs)

        cache = os.path.join(c.tile_cache_dir, _fname(prov)) if c.tile_cache_dir else None
        tmp = tempfile.mkdtemp(prefix="cropgrowth_")
        try:
            done = dp.build_province_datacube_tiled(
                bbox, pm, c.planting_year(pm), tile_deg=c.effective_tile_deg(), geometry=sub,
                cache_dir=cache, tile_workers=c.tile_workers, window=win,
                cache_tag=pl._cfg_tag(pcfg), per_tile_fn=lambda ds: ph.run_phenology(ds, **pcfg),
                tile_sink=sink)
            if done is None:
                return None
            out = []
            for p in periods:
                local = os.path.join(tmp, os.path.basename(self._stage_rel(prov, p)))
                writers[p.label].to_cog(local)
                uri = self.store.put(local, self._stage_rel(prov, p))
                summ = pl.summary_from_counts(pl.stage_counts_from_raster(local))
                csv = os.path.join(tmp, os.path.basename(self._summary_rel(prov, p)))
                summ.to_csv(csv, index=False)
                self.store.put(csv, self._summary_rel(prov, p))
                os.remove(local)
                out.append({"period": p.label, "uri": uri,
                            "hectares": {r["name"]: r["hectares"] for _, r in summ.iterrows()
                                         if r["hectares"] > 0}})
            if qa is not None:
                local = os.path.join(tmp, os.path.basename(self._qa_rel(prov)))
                qa.to_cog(local)
                self.store.put(local, self._qa_rel(prov))
            return out
        finally:
            for wr in all_writers:
                wr.discard()
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------ mosaics
    def _mosaic_rel(self, scope, label, ext=".tiff"):
        c = self.cfg
        return f"{c.prefix()}/mosaic/{c.label()}_{_fname(scope)}_{label}{ext}"

    @staticmethod
    def _scope_name(level, names):
        if level == "national":
            return "PHILIPPINES"
        names = [_key(n) for n in (names or [])]
        return "_".join(names) if 0 < len(names) <= 3 else f"{level.upper()}_{len(names)}"

    def mosaic_periodic(self, level="national", names=None, as_of=None, periods=None, scope=None):
        """
        Merge the saved province stage maps into one COG per period for the selection
        (national -> PHILIPPINES), plus its hectares-per-stage CSV, under <prefix>/mosaic/.
        periods: labels like '202609' / '202609H1'; default: the periods a run on as_of makes.
        Reads province COGs block by block (works for the whole country at 30 m).
        Returns {'mosaics': [...], 'missing': {period: [provinces]}}.
        """
        c = self.cfg
        as_of = self._as_of(as_of)
        units = self.select(level, names)
        provs = sorted(units[c.prov_col])
        scope = scope or self._scope_name(level, names)
        expected = {}                                     # period label -> provinces in season
        for prov in provs:
            for p in self.periods_for(self._planting(units, prov)["window"], as_of):
                expected.setdefault(p.label, []).append(prov)
        labels = list(periods) if periods else sorted(expected)
        self.store.refresh(c.prefix() + "/")
        out = {"scope": scope, "mosaics": [], "missing": {}}
        for label in labels:
            p = pl.Period(None, None, label)
            have = [prov for prov in provs if self.store.exists(self._stage_rel(prov, p))]
            missing = sorted(set(expected.get(label, [])) - set(have))
            if missing:
                out["missing"][label] = missing
            if not have:
                self.log(f"  mosaic {label}: no province maps saved yet")
                continue
            tmp = tempfile.mkdtemp(prefix="cropgrowth_mosaic_")
            try:
                local_in = [self.store.get(self._stage_rel(prov, p), tmp) for prov in have]
                local_out = os.path.join(tmp, os.path.basename(self._mosaic_rel(scope, label)))
                self.log(f"  mosaic {label}: {len(have)} province map(s) -> {scope}")
                info = mosaic_cogs(local_in, local_out, nodata=ph.STAGE_NODATA, log=self.log)
                uri = self.store.put(local_out, self._mosaic_rel(scope, label))
                summ = pl.summary_from_counts(pl.stage_counts_from_raster(local_out))
                csv = os.path.join(tmp, "summary.csv")
                summ.to_csv(csv, index=False)
                self.store.put(csv, self._mosaic_rel(scope, label, "_summary.csv"))
                out["mosaics"].append({
                    "period": label, "uri": uri, "provinces": len(have),
                    "size_px": [info["width"], info["height"]],
                    "hectares": {r["name"]: r["hectares"] for _, r in summ.iterrows() if r["hectares"] > 0}})
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return out

    def run_periodic(self, level="national", names=None, as_of=None, skip_existing=True,
                     start_from=None, only=None, mosaic=None):
        """
        Process the selected provinces and save one stage map per period.
        Provinces are streamed to disk tile by tile (stream_province), so any size fits in memory.
        only   : restrict to these province names (e.g. retrying failures).
        mosaic : afterwards merge the provinces into one COG per period (mosaic_periodic);
                 default cfg.mosaic_after, and only for runs over more than one province.
        Returns {'success': [...], 'skipped', 'out_of_season', 'no_coverage', 'failed'}.
        """
        from .data_processing import tqdm
        as_of = self._as_of(as_of)
        units = self.select(level, names)
        provs = sorted(units[self.cfg.prov_col])
        if only:
            want = {_key(n) for n in only}
            provs = [p for p in provs if p in want]
        if start_from is not None:
            if _key(start_from) not in provs:
                raise ValueError(f"{start_from!r} not in the selection")
            provs = provs[provs.index(_key(start_from)):]
        res = {"as_of": str(as_of.date()), "success": [], "skipped": [], "out_of_season": [],
               "no_coverage": [], "failed": []}
        if skip_existing:
            self.store.refresh(self.cfg.prefix() + "/")
        for prov in tqdm(provs, desc="provinces", unit="prov"):
            ps = self.periods_for(self._planting(units, prov)["window"], as_of)
            if not ps:
                res["out_of_season"].append(prov); continue
            if skip_existing and all(self.store.exists(self._stage_rel(prov, p)) for p in ps):
                res["skipped"].append(prov); self.log(f"  {prov}: skipped (exists)"); continue
            t0 = time.time()
            try:
                maps = self.stream_province(units, prov, as_of, ps)
                if maps is None:
                    res["no_coverage"].append(prov); self.log(f"  {prov}: no coverage"); continue
                res["success"].append({"province": prov, "minutes": round((time.time() - t0) / 60, 1),
                                       "maps": maps})
                self.log(f"  {prov}: ok ({(time.time() - t0) / 60:.1f} min) -> {len(maps)} map(s)")
            except Exception as e:
                res["failed"].append({"province": prov, "error": repr(e)[:500]})
                self.log(f"  {prov}: ERROR {e!r}")
                traceback.print_exc()
            finally:
                gc.collect()
        self.log(" | ".join(f"{k} {len(v)}" for k, v in res.items() if isinstance(v, list)))
        do_mosaic = self.cfg.mosaic_after if mosaic is None else mosaic
        if do_mosaic and len(units) > 1 and (res["success"] or res["skipped"]) and not only:
            try:
                res["mosaic"] = self.mosaic_periodic(level, names, as_of)
            except Exception as e:                              # maps are saved; report and go on
                res["mosaic"] = {"error": repr(e)[:500]}
                self.log(f"  mosaic: ERROR {e!r}")
                traceback.print_exc()
        self._save_run_log(res, "periodic")
        self.last_periodic = res
        return res

    def _save_run_log(self, res, kind):
        rows = ([{"area": s.get("province") or s.get("area"), "status": "success",
                  "outputs": "; ".join(m["uri"] for m in s.get("maps", [])) or s.get("uri")}
                 for s in res["success"]]
                + [{"area": n, "status": k} for k in ("skipped", "out_of_season", "no_coverage")
                   for n in res.get(k, [])]
                + [{"area": f["province"] if "province" in f else f["area"], "status": "failed",
                    "error": f["error"]} for f in res["failed"]])
        if not rows:
            return
        prefix = self.cfg.prefix() if kind == "periodic" else self.cfg.recent_prefix
        stamp = pd.Timestamp.now().strftime("%Y%m%dT%H%M%S")
        tmp = tempfile.mkdtemp(prefix="cropgrowth_")
        try:
            local = os.path.join(tmp, f"{kind}_{stamp}.csv")
            pd.DataFrame(rows).to_csv(local, index=False)
            self.store.put(local, f"{prefix}/runs/{kind}_{stamp}.csv")
        except Exception as e:                                   # a log must not fail a run
            self.log(f"  (run log not saved: {e!r})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------ recent
    def run_recent(self, names=None, bbox=None, as_of=None, lookback_days=None):
        """
        Current growth stage for municipal / barangay areas (`names` from the
        AOI file) or one bbox (w, s, e, n). Saves a 3-band COG + hectares CSV
        per area. Returns {'success': [...], 'failed': [...], 'no_coverage': [...]}.
        """
        c = self.cfg
        as_of = self._as_of(as_of)
        lookback = int(lookback_days or c.lookback_days)
        areas = []
        if bbox is not None:
            areas.append(("BBOX_" + "_".join(f"{v:.3f}" for v in bbox), tuple(bbox)))
        if names:
            g = self.aoi_gdf()
            for n in names:
                sub = g[g[c.aoi_name_col] == _key(n)]
                areas.append((_key(n), sub if not sub.empty else None))
        res = {"as_of": str(as_of.date()), "success": [], "no_coverage": [], "failed": []}
        for name, area in areas:
            if area is None:
                res["failed"].append({"area": name, "error": f"not found in {c.aoi_name_col!r}"}); continue
            try:
                cache = os.path.join(c.tile_cache_dir, "recent", _fname(name)) if c.tile_cache_dir else None
                rec, _ = pl.run_recent(area, as_of, lookback, tile_deg=c.effective_tile_deg(),
                                       pheno_cfg=c.pheno_cfg, cache_dir=cache,
                                       tile_workers=c.tile_workers,
                                       date_median_radius=c.date_median_radius)
                if rec is None:
                    res["no_coverage"].append(name); continue
                base = f"{c.recent_prefix}/{_fname(name)}_{rec.attrs['as_of'].replace('-', '')}"
                summ = pl.stage_area_summary(rec["growth_stage"])
                tmp = tempfile.mkdtemp(prefix="cropgrowth_")
                try:
                    tif = os.path.join(tmp, "recent.tiff"); ph.write_recent_cog(rec, tif)
                    uri = self.store.put(tif, base + ".tiff")
                    csv = os.path.join(tmp, "summary.csv"); summ.to_csv(csv, index=False)
                    self.store.put(csv, base + "_summary.csv")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                age = rec["data_age_days"].values
                res["success"].append({
                    "area": name, "uri": uri, "as_of": rec.attrs["as_of"],
                    "data_age_days_median": None if np.isnan(age).all() else float(np.nanmedian(age)),
                    "hectares": {r["name"]: r["hectares"] for _, r in summ.iterrows() if r["hectares"] > 0},
                    "share_pct": {r["name"]: r["share_%"] for _, r in summ.iterrows() if r["hectares"] > 0},
                    "_data": rec})
                self.log(f"  {name}: ok -> {uri}")
            except Exception as e:
                res["failed"].append({"area": name, "error": repr(e)[:500]})
                self.log(f"  {name}: ERROR {e!r}")
                traceback.print_exc()
            finally:
                gc.collect()
        self._save_run_log(res, "recent")
        self.last_recent = res
        return res

    # ------------------------------------------------------------ helpers
    def quick_check(self, bbox, planting_month=None, year=None):
        """Phenology on a small bbox: QC outcome counts (for tuning pheno_cfg).
        planting_month: default the season's fallback; year: planting year (default
        from the season)."""
        c = self.cfg
        pm = planting_month or c.fallback_month()
        ds = dp.build_province_datacube(tuple(bbox), pm, year or c.planting_year(pm))
        if ds is None:
            return {"status": "no coverage / no cropland in bbox"}
        p = ph.run_phenology(ds, **c.pheno_cfg)
        qc = p["qc"].values
        qc = qc[np.isfinite(qc)].astype(int)
        counts = pd.Series(qc).map(ph.QC_DESCRIPTION).value_counts()
        return {"crop_pixels": int(qc.size), "rice_cycle_pixels": int((qc < 10).sum()),
                "qc_counts": {k: int(v) for k, v in counts.items()}}

    def list_outputs(self, kind="periodic", contains=None, limit=200):
        prefix = self.cfg.prefix() if kind == "periodic" else self.cfg.recent_prefix
        names = self.store.list(prefix + "/")
        if contains:
            names = [n for n in names if _key(contains) in _key(n)]
        return {"store": self.store.describe(), "count": len(names), "files": names[-limit:]}

    def read_summary(self, rel_path):
        """A saved hectares-per-stage CSV as records."""
        import io
        return pd.read_csv(io.StringIO(self.store.read_text(rel_path))).to_dict("records")
