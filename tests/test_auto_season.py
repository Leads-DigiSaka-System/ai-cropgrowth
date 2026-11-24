"""Auto season: planting months from both columns, up to two crop cycles per pixel."""
import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent import phenology as ph
from cropgrowth_agent.runner import RunConfig, Runner


def _two_crop_field(end="2026-10-15"):
    t = pd.date_range("2025-11-01", "2026-10-31", freq="10D")
    g = lambda c, w: np.exp(-0.5 * (((t - pd.Timestamp(c)).days.values) / w) ** 2)
    v = 0.18 + 0.65 * g("2026-03-01", 24) + 0.62 * g("2026-08-25", 22)       # dry crop + wet crop
    v[t > pd.Timestamp(end)] = np.nan
    ds = xr.DataArray(v[:, None, None].astype("float32"), dims=("time", "y", "x"),
                      coords={"time": t, "y": [10.0], "x": [120.0]}).to_dataset(name="NDVI")
    ds.attrs.update(spatial_dims=["y", "x"], window_start="2025-11-01")
    return ds


def test_two_cycles_found_and_each_month_classified():
    out = ph.run_phenology(_two_crop_field(), max_cycles=2, PEAK_SELECT="last", PEAK_SEARCH_DAYS=(30, 365))
    day = lambda b: ph.from_epoch_days(float(out[b][0, 0]))[0]
    assert int(out["qc"][0, 0]) == ph.QC_COMPLETE and int(out["prev_qc"][0, 0]) == ph.QC_COMPLETE
    assert pd.Timestamp("2026-02-20") < day("prev_peak") < pd.Timestamp("2026-03-10")
    assert pd.Timestamp("2026-08-15") < day("peak") < pd.Timestamp("2026-09-05")
    assert day("prev_harvest") <= day("emergence")
    stages = {m: int(ph.classify_stage_month(out, m).values[0, 0])
              for m in ["2026-02", "2026-05", "2026-08", "2026-10"]}
    assert stages == {"2026-02": 3, "2026-05": 5, "2026-08": 3, "2026-10": 5}
    # one cycle only sees the wet crop and calls February "pre-planting"
    one = ph.run_phenology(_two_crop_field(), PEAK_SELECT="last", PEAK_SEARCH_DAYS=(30, 365))
    assert int(ph.classify_stage_month(one, "2026-02").values[0, 0]) == 1


def test_single_cycle_outputs_unchanged_by_default():
    out = ph.run_phenology(_two_crop_field())
    assert not any(v.startswith(ph.PREV) for v in out.data_vars)


def test_auto_is_the_default():
    c = RunConfig()
    assert c.is_auto() and c.plant_col() is None and c.label() == "rice"
    assert c.prefix() == "products/growth_stage/auto"
    assert not RunConfig(plant_mo_col="Semester_2").is_auto()
    assert RunConfig(season="auto", plant_mo_col="Semester_1").is_auto()


@pytest.fixture
def make_runner(mpc, boundaries, tmp_path):
    def make(**kw):
        cfg = RunConfig(resolution_m=dp.GRID_SCALE_DEG * dp.M_PER_DEG, tile_deg=0.1, tile_workers=1,
                        date_median_radius=0, vector_path=boundaries["provinces"], region_col="Reg_Name",
                        output_target="local", local_root=str(tmp_path / "out"), **kw)
        return Runner(cfg, log=lambda *a: None)
    return make


def test_auto_plan_uses_both_columns(make_runner):
    r = make_runner()
    # Alpha: Semester_1 = 12 (dry), Semester_2 = 6 (wet)
    dry_time = {d["province"]: d for d in r.plan_periodic("regional", ["R1"], "2026-04-01")["detail"]}
    assert dry_time["ALPHA"]["periods"] == ["202603"]
    assert dry_time["ALPHA"]["window"] == "2025-11-01..2026-06-30"       # the dry crop's window
    wet_time = {d["province"]: d for d in r.plan_periodic("regional", ["R1"], "2026-10-01")["detail"]}
    assert wet_time["ALPHA"]["periods"] == ["202609"]
    assert wet_time["ALPHA"]["window"].startswith("2026-05-01")           # the wet crop's window
    assert wet_time["ALPHA"]["planting_months"] == {"Semester_1": {12: 1}, "Semester_2": {6: 1}}
    # Beta has no Semester_2 value: its dry month still counts; no wet candidate
    assert wet_time["BETA"]["planting_months"] == {"Semester_1": {12: 1}}


def test_overlapping_seasons_share_one_window(make_runner):
    r = make_runner(periods="season")
    plan = {d["province"]: d for d in r.plan_periodic("provincial", ["Alpha"], "2026-05-10")["detail"]}
    # backfill to 10 May: periods Nov 2025 - Apr 2026. November can still hold the wet-2025 crop
    # (window May-Dec 2025) and the rest is the dry-2026 crop (Nov-Jun): one window over both crops
    a = plan["ALPHA"]
    assert a["window"] == "2025-05-01..2026-06-30"
    assert a["periods"][0] == "202511" and a["periods"][-1] == "202604"


def test_auto_run_writes_rice_maps(make_runner):
    r = make_runner(save_phenology_bands=True)
    res = r.run_periodic("provincial", ["Alpha"], "2026-04-01")
    m = res["success"][0]["maps"][0]
    assert os.path.basename(m["uri"]) == "rice_ALPHA_202603.tiff"
    assert "/products/growth_stage/auto/" in m["uri"] and m["hectares"]
