import os

import pytest

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent.runner import RunConfig, Runner


@pytest.fixture
def runner(mpc, boundaries, tmp_path):
    cfg = RunConfig(data_source="hls", resolution_m=dp.GRID_SCALE_DEG * dp.M_PER_DEG, tile_deg=0.1,
                    tile_workers=1, date_median_radius=0, vector_path=boundaries["provinces"],
                    region_col="Reg_Name", aoi_path=boundaries["aoi"], output_target="local", season_year=2026,
                    local_root=str(tmp_path / "outputs"))
    return Runner(cfg, log=lambda *a: None)


def test_areas_and_plan(runner):
    assert runner.list_areas("province") == ["ALPHA", "BETA", "PALAWAN"]
    assert runner.list_areas("region") == ["R1", "R4"]
    plan = runner.plan_periodic("regional", ["r1"], "2026-04-01")
    assert plan["provinces"] == 2 and plan["detail"][0]["periods"] == ["202603"]
    assert "PALAWAN" in list(runner.select("national")["Pro_Name"])          # no default exclusion
    runner.update(exclude=("PALAWAN",))
    assert "PALAWAN" not in list(runner.select("national")["Pro_Name"])


def test_periodic_run_saves_maps_summaries_and_log(runner):
    res = runner.run_periodic("regional", ["R1"], "2026-04-01")
    assert [s["province"] for s in res["success"]] == ["ALPHA", "BETA"]
    m = res["success"][0]["maps"][0]
    assert m["period"] == "202603" and os.path.exists(m["uri"]) and m["hectares"]
    files = runner.list_outputs("periodic")["files"]
    assert any("/summary/" in f for f in files) and any("/runs/periodic_" in f for f in files)
    rows = runner.read_summary(next(f for f in files if "/summary/" in f))
    assert {r["name"] for r in rows} >= {"vegetative", "reproductive"}

    assert runner.run_periodic("regional", ["R1"], "2026-04-01")["skipped"] == ["ALPHA", "BETA"]
    only = runner.run_periodic("national", None, "2026-04-01", skip_existing=False, only=["beta"])
    assert [s["province"] for s in only["success"]] == ["BETA"]


def test_recent_run(runner):
    res = runner.run_recent(["muñoz town"], as_of="2026-03-20")
    s = res["success"][0]
    assert list(s["hectares"]) == ["vegetative"] and os.path.exists(s["uri"])
    assert s["data_age_days_median"] <= 5
    assert runner.run_recent(["nowhere"])["failed"][0]["error"].startswith("not found")


def test_update_is_validated(runner, tmp_path):
    runner.update(local_root=str(tmp_path / "o2"), pheno_cfg={"MIN_AMPLITUDE": 0.25})
    assert runner.store.root.endswith("o2")
    assert runner.cfg.pheno_cfg["MIN_AMPLITUDE"] == 0.25 and runner.cfg.pheno_cfg["SG_WINDOW"] == 5
    for bad in ({"cadence": "weekly"}, {"nope": 1}, {"pheno_cfg": {"BAD": 1}}):
        with pytest.raises(ValueError):
            runner.update(**bad)
