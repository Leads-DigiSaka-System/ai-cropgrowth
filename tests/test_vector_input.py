"""Reading the province table: multi-layer GeoPackages, column-name variants and
municipal-level rows (several planting months per province)."""
import pytest

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent.runner import RunConfig, Runner, read_vector


def _write(path, layer, **cols):
    n = len(next(iter(cols.values())))
    g = gpd.GeoDataFrame(cols, geometry=[box(120.0 + 0.01 * i, 10.0, 120.01 + 0.01 * i, 10.01) for i in range(n)],
                         crs=4326)
    g.to_file(path, layer=layer, driver="GPKG")


def _runner(path, tmp_path, **kw):
    cfg = RunConfig(vector_path=str(path), season_year=2026, output_target="local",
                    local_root=str(tmp_path / "o"), **kw)
    return Runner(cfg, log=lambda *a: None)


def test_finds_the_layer_with_the_column(tmp_path):
    f = tmp_path / "admin.gpkg"
    _write(f, "municipal", Pro_Name=["BOHOL"], Mun_Name=["TAGBILARAN"], Semester_1=[11])          # first layer
    _write(f, "province", Pro_Name=["BOHOL"], Semester_1=[11], Semester_2=[6])
    r = _runner(f, tmp_path, plant_mo_col="Semester_2")
    g = r.provinces_gdf()
    assert "Semester_2" in g and "Mun_Name" not in g
    r.update(plant_mo_col="Semester_1", vector_layer="municipal")
    assert "Mun_Name" in r.provinces_gdf()


def test_column_name_variants(tmp_path):
    f = tmp_path / "p.gpkg"
    _write(f, "p", PRO_NAME=["BOHOL"], **{"semester 2": [6]})
    g = read_vector(str(f), ["Pro_Name", "Semester_2"], log=lambda *a: None)
    assert {"Pro_Name", "Semester_2"} <= set(g.columns)


def test_missing_column_error_lists_layers_and_drive_hint(tmp_path):
    f = tmp_path / "p.gpkg"
    _write(f, "municipal", Pro_Name=["BOHOL"], Semester_1=[11])
    r = _runner(f, tmp_path, plant_mo_col="Semester_2")
    with pytest.raises(ValueError) as e:
        r.provinces_gdf()
    msg = str(e.value)
    assert "Semester_2" in msg and "'municipal'" in msg and "force_remount" in msg


def test_municipal_rows_give_one_window_covering_all_planting_months(tmp_path):
    f = tmp_path / "mun.gpkg"
    months = [10, 11, 11, 12, 12, 12, 11, 10, 11, 12] * 2 + [5]          # one outlier row (May)
    _write(f, "mun", Pro_Name=["BOHOL"] * len(months), Mun_Name=[f"M{i}" for i in range(len(months))],
           Semester_1=months)
    r = _runner(f, tmp_path, plant_mo_col="Semester_1")
    info = r._planting(r.provinces_gdf(), "BOHOL")
    assert info["pm"] == 10
    assert info["window"] == ("2025-09-01", "2026-06-30")       # Oct 2025 - 1 month .. Dec 2025 + 6 months
    assert "[5]" in info["note"]                                # rare month ignored and reported
    plan = r.plan_periodic("provincial", ["Bohol"], "2026-04-01")
    assert plan["detail"][0]["planting_months"] == {5: 1, 10: 4, 11: 8, 12: 8}
    assert plan["detail"][0]["periods"] == ["202603"]
