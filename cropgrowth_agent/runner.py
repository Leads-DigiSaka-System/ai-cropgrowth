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
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd

from . import data_processing as dp
from . import phenology as ph
from . import pipeline as pl
from . import output_store as storage

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
    plant_mo_col: str = "Semester_1"
    exclude: tuple = ("PALAWAN",)
    year: int = 2025
    planting_month_fallback: int = 12
    cadence: str = "monthly"               # 'monthly' | 'semimonthly'
    periods: str = "last_complete"         # 'last_complete' | 'current' | 'season'
    stage_method: str = "dominant"         # 'dominant' | 'midpoint'
    save_phenology_bands: bool = False
    cog_label: str = "dry2026"
    output_prefix: str = "products/growth_stage/2026/dry"

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
        checks = {"data_source": ("hls", "s2"), "cadence": pl.CADENCES,
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
        for k, v in changes.items():
            setattr(self.cfg, k, v)
        if {"output_target", "gcs_bucket", "drive_root", "local_root"} & set(changes):
            self._store = None
            self.store                                       # fail now, not mid-run
        if {"vector_path", "prov_col"} & set(changes):
            self._prov_gdf = None
        if {"aoi_path", "aoi_name_col"} & set(changes):
            self._aoi_gdf = None
        self.cfg.apply()
        return self.cfg.public()

    def provinces_gdf(self):
        if self._prov_gdf is None:
            import geopandas as gpd
            if not self.cfg.vector_path:
                raise ValueError("vector_path (province boundaries) is not set")
            g = gpd.read_file(self.cfg.vector_path).to_crs(4326)
            g[self.cfg.prov_col] = g[self.cfg.prov_col].map(_key)
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
    def _pm(self, units, prov):
        v = units.loc[units[self.cfg.prov_col] == prov].iloc[0][self.cfg.plant_mo_col]
        return int(v) if pd.notna(v) else self.cfg.planting_month_fallback

    def periods_for(self, pm, as_of):
        c = self.cfg
        s, e = dp.season_window(pm, c.year)
        season = pl.periods_between(s, e, c.cadence)
        if c.periods == "season":
            ps = pl.season_periods(pm, c.year, c.cadence, until=as_of)
        elif c.periods == "current":
            ps = [pl.period_of(as_of, c.cadence)]
        else:
            ps = [pl.last_complete_period(as_of, c.cadence)]
        return [p for p in ps if p in season]

    def _stage_rel(self, prov, period):
        c = self.cfg
        return f"{c.output_prefix}/{c.cog_label}_{_fname(prov)}_{period.label}.tiff"

    def _summary_rel(self, prov, period):
        c = self.cfg
        return f"{c.output_prefix}/summary/{c.cog_label}_{_fname(prov)}_{period.label}.csv"

    def _qa_rel(self, prov):
        c = self.cfg
        return f"{c.output_prefix}/qa/{c.cog_label}_{_fname(prov)}_phenology_dates.tiff"

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
            pm = self._pm(units, prov)
            ps = self.periods_for(pm, as_of)
            nt = len(dp.generate_tiles(tuple(sub.total_bounds), tile, geometry=sub))
            n_tiles += nt if ps else 0
            rows.append({"province": prov, "planting_month": pm, "tiles": nt,
                         "periods": [p.label for p in ps]})
        return {"level": level, "names": names or [], "as_of": str(as_of.date()),
                "provinces": len(rows), "provinces_in_season": sum(bool(r["periods"]) for r in rows),
                "tiles_to_process": n_tiles, "tile_deg": tile,
                "data_source": self.cfg.data_source, "resolution_m": self.cfg.resolution_m,
                "cadence": self.cfg.cadence, "periods_mode": self.cfg.periods,
                "output": f"{self.store.describe()}/{self.cfg.output_prefix}",
                "detail": rows}

    def process_province(self, units, prov, as_of):
        c = self.cfg
        sub = units.loc[units[c.prov_col] == prov]
        cache = os.path.join(c.tile_cache_dir, _fname(prov)) if c.tile_cache_dir else None
        pheno = pl.run_periodic_unit(sub, self._pm(units, prov), c.year, as_of,
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
                ph.write_cog(pheno, local, ref_year=c.year)
                self.store.put(local, self._qa_rel(prov))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return out

    def run_periodic(self, level="national", names=None, as_of=None, skip_existing=True,
                     start_from=None, only=None):
        """
        Process the selected provinces and save one stage map per period.
        only : restrict to these province names (e.g. retrying failures).
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
            self.store.refresh(self.cfg.output_prefix + "/")
        for prov in tqdm(provs, desc="provinces", unit="prov"):
            ps = self.periods_for(self._pm(units, prov), as_of)
            if not ps:
                res["out_of_season"].append(prov); continue
            if skip_existing and all(self.store.exists(self._stage_rel(prov, p)) for p in ps):
                res["skipped"].append(prov); self.log(f"  {prov}: skipped (exists)"); continue
            t0 = time.time()
            try:
                pheno = self.process_province(units, prov, as_of)
                if pheno is None:
                    res["no_coverage"].append(prov); self.log(f"  {prov}: no coverage"); continue
                maps = self.save_periodic(pheno, prov, ps)
                res["success"].append({"province": prov, "minutes": round((time.time() - t0) / 60, 1),
                                       "maps": maps})
                self.log(f"  {prov}: ok ({(time.time() - t0) / 60:.1f} min) -> {len(maps)} map(s)")
                del pheno
            except Exception as e:
                res["failed"].append({"province": prov, "error": repr(e)[:500]})
                self.log(f"  {prov}: ERROR {e!r}")
                traceback.print_exc()
            finally:
                gc.collect()
        self.log(" | ".join(f"{k} {len(v)}" for k, v in res.items() if isinstance(v, list)))
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
        prefix = self.cfg.output_prefix if kind == "periodic" else self.cfg.recent_prefix
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
        """Phenology on a small bbox: QC outcome counts (for tuning pheno_cfg)."""
        c = self.cfg
        ds = dp.build_province_datacube(tuple(bbox), planting_month or c.planting_month_fallback,
                                        year or c.year)
        if ds is None:
            return {"status": "no coverage / no cropland in bbox"}
        p = ph.run_phenology(ds, **c.pheno_cfg)
        qc = p["qc"].values
        qc = qc[np.isfinite(qc)].astype(int)
        counts = pd.Series(qc).map(ph.QC_DESCRIPTION).value_counts()
        return {"crop_pixels": int(qc.size), "rice_cycle_pixels": int((qc < 10).sum()),
                "qc_counts": {k: int(v) for k, v in counts.items()}}

    def list_outputs(self, kind="periodic", contains=None, limit=200):
        prefix = self.cfg.output_prefix if kind == "periodic" else self.cfg.recent_prefix
        names = self.store.list(prefix + "/")
        if contains:
            names = [n for n in names if _key(contains) in _key(n)]
        return {"store": self.store.describe(), "count": len(names), "files": names[-limit:]}

    def read_summary(self, rel_path):
        """A saved hectares-per-stage CSV as records."""
        import io
        return pd.read_csv(io.StringIO(self.store.read_text(rel_path))).to_dict("records")
