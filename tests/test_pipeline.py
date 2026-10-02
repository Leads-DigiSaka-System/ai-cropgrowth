import numpy as np
import pandas as pd
import pytest

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent import pipeline as pl


def test_periods():
    assert pl.period_of("2026-02-10", "semimonthly").label == "202602H1"
    assert pl.period_of("2026-02-16", "semimonthly").label == "202602H2"
    assert pl.last_complete_period("2026-03-01").label == "202602"
    assert pl.last_complete_period("2026-02-28").label == "202602"
    assert pl.last_complete_period("2026-02-16", "semimonthly").label == "202602H1"
    assert [p.label for p in pl.periods_between("2025-11-01", "2026-01-31")] == ["202511", "202512", "202601"]
    assert len(pl.season_periods(12, 2025, "semimonthly")) == 16
    assert [p.label for p in pl.season_periods(12, 2025, until="2026-01-31")] == ["202511", "202512", "202601"]


def test_select_units():
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"Pro_Name": ["BOHOL", "CEBU", "PALAWAN", "ILOILO"],
                          "Reg_Name": ["VII", "VII", "IV-B", "VI"]}, geometry=[box(0, 0, 1, 1)] * 4, crs=4326)
    assert list(pl.select_units(g, "national", exclude=["PALAWAN"]).Pro_Name) == ["BOHOL", "CEBU", "ILOILO"]
    assert list(pl.select_units(g, "regional", ["vii"], region_col="Reg_Name").Pro_Name) == ["BOHOL", "CEBU"]
    assert list(pl.select_units(g, "provincial", ["Iloilo"]).Pro_Name) == ["ILOILO"]
    with pytest.raises(ValueError):
        pl.select_units(g, "provincial", ["NOPE"])


def test_run_recent_follows_latest_cycle(mpc):
    dp.set_data_source("hls")
    rec, pheno = pl.run_recent((120.0, 10.0, 120.2, 10.1), "2026-03-20", 240, tile_deg=0.1, date_median_radius=0)
    st = rec["growth_stage"].values
    assert (st == 2).all()                         # young crop: vegetative, not the old harvested one
    assert np.nanmax(rec["data_age_days"].values) <= 5
    assert rec.attrs["data_source"] == "hls"


def test_periodic_window_clipped_and_cached_per_as_of(mpc, tmp_path):
    import os
    from shapely.geometry import box
    dp.set_data_source("s2")
    area = box(120.0, 10.0, 120.2, 10.1)
    p1 = pl.run_periodic_unit(area, 12, 2025, "2026-02-01", tile_deg=0.1, cache_dir=str(tmp_path),
                              date_median_radius=0)
    assert p1.attrs["window_end"] == "2026-02-01" and p1.attrs["season_end"] == "2026-06-30"
    n1 = len(os.listdir(tmp_path))
    pl.run_periodic_unit(area, 12, 2025, "2026-03-01", tile_deg=0.1, cache_dir=str(tmp_path), date_median_radius=0)
    assert len(os.listdir(tmp_path)) == 2 * n1     # a new as_of never reuses the old tiles
    assert pl.run_periodic_unit(area, 12, 2025, "2025-10-01") is None   # season not started


def test_stage_area_summary_hectares():
    import xarray as xr
    res = dp.GRID_SCALE_DEG
    a = np.full((200, 200), 2, np.int16)
    a[:50] = 3
    a[:, :10] = -1
    st = xr.DataArray(a, dims=("y", "x"),
                      coords={"y": 10.0 - res * np.arange(200), "x": 120.0 + res * np.arange(200)})
    df = pl.stage_area_summary(st).set_index("name")
    px = (res * dp.M_PER_DEG) ** 2 * np.cos(np.radians(10.0 - res * 100)) / 1e4
    assert df.loc["vegetative", "pixels"] == 150 * 190 and df.loc["reproductive", "pixels"] == 50 * 190
    assert df.loc["vegetative", "hectares"] == pytest.approx(150 * 190 * px, rel=0.01)
    assert df.loc["vegetative", "share_%"] == pytest.approx(75, abs=0.1)
