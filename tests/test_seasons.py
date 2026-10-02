import os

import pytest

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent import pipeline as pl
from cropgrowth_agent.runner import RunConfig, Runner


@pytest.mark.parametrize("value,month", [
    (5, 5), (5.0, 5), ("5", 5), ("05", 5), (" May", 5), ("MAY", 5), ("June", 6), ("5-6", 5),
    ("May-June", 5), ("12.0", 12), ("Sept", 9), (0, None), (None, None), (float("nan"), None),
    ("n/a", None), (13, None)])
def test_parse_month(value, month):
    assert pl.parse_month(value) == month


def test_season_from_column():
    c = RunConfig(plant_mo_col="Semester_2", season_year=2026)
    assert c.season_name() == "wet" and c.fallback_month() == 6
    assert c.label() == "wet2026" and c.prefix() == "products/growth_stage/2026/wet"
    d = RunConfig(plant_mo_col="Semester_1", season_year=2026)
    assert d.season_name() == "dry" and d.fallback_month() == 12 and d.label() == "dry2026"
    assert RunConfig(season="wet").plant_col() == "Semester_2"
    with pytest.raises(ValueError, match="dry season's column"):
        RunConfig(plant_mo_col="Semester_1", season="wet").season_name()


def test_season_windows():
    wet = RunConfig(plant_mo_col="Semester_2", season_year=2026)
    assert dp.season_window(6, wet.planting_year(6)) == ("2026-05-01", "2026-12-31")
    dry = RunConfig(plant_mo_col="Semester_1", season_year=2026)
    assert dp.season_window(11, dry.planting_year(11)) == ("2025-10-01", "2026-05-31")
    assert dp.season_window(12, dry.planting_year(12)) == ("2025-11-01", "2026-06-30")
    # a January dry-season planting is January of the harvest year, not of the year before
    assert dp.season_window(1, dry.planting_year(1)) == ("2025-12-01", "2026-07-31")


def test_legacy_year_and_defaults():
    assert RunConfig(plant_mo_col="Semester_1", year=2025).label() == "dry2026"   # old notebook setting
    assert RunConfig(plant_mo_col="Semester_2", year=2026).label() == "wet2026"
    assert pl.default_season_year("dry", "2026-10-02") == 2027                    # dry2027 planting now
    assert pl.default_season_year("wet", "2026-10-02") == 2026


@pytest.fixture
def make_runner(mpc, boundaries, tmp_path):
    def make(**kw):
        cfg = RunConfig(resolution_m=dp.GRID_SCALE_DEG * dp.M_PER_DEG, tile_deg=0.1, tile_workers=1,
                        date_median_radius=0, vector_path=boundaries["provinces"], region_col="Reg_Name",
                        output_target="local", local_root=str(tmp_path / "out"), **kw)
        return Runner(cfg, log=lambda *a: None)
    return make


def test_wet_season_plan_uses_semester_2(make_runner):
    r = make_runner(plant_mo_col="Semester_2", season_year=2026)
    plan = r.plan_periodic("regional", ["R1"], "2026-10-01")
    assert plan["season"] == "wet2026" and plan["planting_month_column"] == "Semester_2"
    rows = {d["province"]: d for d in plan["detail"]}
    assert rows["ALPHA"]["planting_month"] == 6 and rows["ALPHA"]["window"] == "2026-05-01..2026-12-31"
    assert rows["BETA"]["planting_month"] == 6 and "fallback" in rows["BETA"]["note"]     # empty cell
    assert rows["ALPHA"]["periods"] == ["202609"]
    assert plan["output"].endswith("products/growth_stage/2026/wet")


def test_dry_season_is_out_of_season_in_october(make_runner):
    r = make_runner(plant_mo_col="Semester_1", season_year=2026)
    plan = r.plan_periodic("regional", ["R1"], "2026-10-01")
    assert all(d["periods"] == [] for d in plan["detail"])      # dry2026 ended in June
    res = r.run_periodic("regional", ["R1"], "2026-10-01")
    assert res["out_of_season"] == ["ALPHA", "BETA"] and not res["success"]


def test_wet_season_run_writes_wet_maps(make_runner):
    r = make_runner(plant_mo_col="Semester_2", season_year=2026, periods="season")
    res = r.run_periodic("provincial", ["Alpha"], "2026-10-01")
    maps = res["success"][0]["maps"]
    assert [m["period"] for m in maps] == ["202605", "202606", "202607", "202608", "202609"]
    assert all(os.path.basename(m["uri"]).startswith("wet2026_ALPHA_") for m in maps)
    assert "/products/growth_stage/2026/wet/" in maps[0]["uri"]


def test_switching_season_follows_column(make_runner):
    r = make_runner(plant_mo_col="Semester_1", season_year=2026)
    r.update(season="wet")
    assert r.cfg.plant_col() == "Semester_2" and r.cfg.label() == "wet2026"
    with pytest.raises(ValueError):
        r.update(plant_mo_col="Semester_1", season="wet")
    assert r.cfg.season_name() == "wet"                          # rejected change rolled back


def test_missing_column_is_explained(make_runner):
    r = make_runner(plant_mo_col="Semester_3", season="wet", season_year=2026)
    with pytest.raises(ValueError, match="Semester_3"):
        r.plan_periodic("national")
