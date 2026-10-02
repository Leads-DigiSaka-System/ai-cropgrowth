import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent import phenology as ph


def test_resolution_and_resampling(monkeypatch):
    monkeypatch.setattr(dp, "GRID_SCALE_DEG", dp.GRID_SCALE_DEG)
    assert dp.set_resolution(30) == pytest.approx(30 / 111_320)
    assert isinstance(dp._s2_resampling(), dict)                  # average / mode when coarse
    dp.set_resolution(10)
    assert dp._s2_resampling() == "nearest"


def test_settings_are_read_at_call_time(monkeypatch):
    monkeypatch.setattr(dp, "INTERVAL_DAYS", 5)
    assert len(dp.anchor_dates("2025-11-01", "2026-06-30")) == 49


def test_end_aligned_anchors():
    a = dp.anchor_dates("2026-01-01", "2026-03-28", 10, align="end")
    assert a[-1] == pd.Timestamp("2026-03-28") and a[0] >= pd.Timestamp("2026-01-01")


def test_tiles_outside_geometry_dropped():
    from shapely.geometry import box
    tiles = dp.generate_tiles((120.0, 10.0, 120.5, 10.5), 0.1, geometry=box(120.0, 10.0, 120.15, 10.15))
    assert len(tiles) == 4                         # edge-touching tiles don't count


def test_ndvi_from_l2a_removes_boa_offset():
    v = dp.ndvi_from_l2a(np.array([1500]), np.array([4000]), np.array([4]), "2025-01-01")
    assert v[0] == pytest.approx((0.3 - 0.05) / (0.3 + 0.05), abs=1e-4)
    assert np.isnan(dp.ndvi_from_l2a(np.array([1500]), np.array([4000]), np.array([9]), "2025-01-01")[0])


def test_ndvi_from_hls_fmask():
    red = np.array([500, 500, 500, 500, 500, -9999, 500, 300], "int16")
    nir = np.array([3000, 3000, 3000, 3000, 3000, 3000, 3000, 200], "int16")
    fm = np.array([0, 1 << 1, 1 << 3, 1 << 2, 3 << 6, 0, 255, 1 << 5], "uint8")
    v = dp.ndvi_from_hls(red, nir, fm)
    assert v[0] == pytest.approx(2500 / 3500)
    assert np.isnan(v[1:7]).all()                  # cloud, shadow, adjacent, aerosol, fill
    assert v[7] == pytest.approx(-0.2)             # water kept (flooded paddy)


def test_bad_scene_dropped_and_remembered(mpc):
    dp.set_data_source("s2")
    mpc.pools["sentinel-2-l2a"].append(type(mpc.pools["sentinel-2-l2a"][0])(
        "sentinel-2-l2a", 1, ["B04", "B08", "SCL"]))
    mpc.bad_ids = {"sentinel-2-l2a_1"}
    cube = dp.load_ndvi_stack((120.0, 10.0, 120.05, 10.05), "2025-11-01", "2026-02-01")
    assert cube is not None and cube.sizes["time"] > 0
    assert any("sentinel-2-l2a_1" in h for h in dp._BAD_HREFS)
    n_loads = len(mpc.loads)
    dp.load_ndvi_stack((120.05, 10.0, 120.1, 10.05), "2025-11-01", "2026-02-01")
    assert len(mpc.loads) == n_loads + 1           # second tile: no failed attempt first


def test_offshore_tile_skips_s2(mpc):
    mpc.pools["esa-worldcover"] = []
    assert dp.build_province_datacube((120, 10, 120.1, 10.1), 12, 2025) is None
    assert mpc.loads == []


def test_cropland_first_single_search_and_shrink(mpc):
    dp.set_data_source("s2")
    mpc.crop_box = (120.0, 10.0, 120.12, 10.08)
    out = dp.build_province_datacube_tiled((120.0, 10.0, 120.4, 10.1), 12, 2025, per_tile_fn=ph.run_phenology)
    assert mpc.searches == ["sentinel-2-l2a", "esa-worldcover"]            # once per province
    assert [b for _, b in mpc.loads] == [(120.0, 10.0, 120.1, 10.08), (120.1, 10.0, 120.12, 10.08)]
    qc = out["qc"].values
    inside = (out.x.values[None, :] < 120.12) & (out.y.values[:, None] < 10.08)
    assert np.isfinite(qc[inside]).all() and not np.isfinite(qc[~inside]).any()


def test_parallel_equals_sequential(mpc):
    dp.set_data_source("s2")
    bb = (120.0, 10.0, 120.3, 10.1)
    a = dp.build_province_datacube_tiled(bb, 12, 2025, per_tile_fn=ph.run_phenology)
    b = dp.build_province_datacube_tiled(bb, 12, 2025, per_tile_fn=ph.run_phenology, tile_workers=3)
    xr.testing.assert_identical(a, b)


def test_tile_cache_resume_and_keys(mpc, tmp_path):
    dp.set_data_source("s2")
    bb = (120.0, 10.0, 120.2, 10.1)
    first = dp.build_province_datacube_tiled(bb, 12, 2025, per_tile_fn=ph.run_phenology, cache_dir=str(tmp_path))
    mpc.searches.clear(); mpc.loads.clear()
    again = dp.build_province_datacube_tiled(bb, 12, 2025, per_tile_fn=ph.run_phenology, cache_dir=str(tmp_path))
    assert mpc.searches == [] and mpc.loads == []
    xr.testing.assert_allclose(first, again)
    k = dp._cache_key(("a", "b"), "start", True)
    assert k != dp._cache_key(("a", "c"), "start", True) != dp._cache_key(("a", "b"), "start", True, "cfg2")


def test_failed_tile_fails_province_but_keeps_cache(mpc, tmp_path, monkeypatch):
    dp.set_data_source("s2")
    real = dp._build_tile

    def flaky(tb, *a, **k):
        if tb[0] > 120.05:
            raise AssertionError("boom")
        return real(tb, *a, **k)
    monkeypatch.setattr(dp, "_build_tile", flaky)
    with pytest.raises(RuntimeError, match="1/2 tile"):
        dp.build_province_datacube_tiled((120.0, 10.0, 120.2, 10.1), 12, 2025, per_tile_fn=ph.run_phenology,
                                         cache_dir=str(tmp_path), tile_retries=1)
    assert len(os.listdir(tmp_path)) == 1


def test_empty_worldcover_search_fails_province(mpc):
    mpc.pools["esa-worldcover"] = []
    with pytest.raises(dp.NoWorldCoverError):
        dp.build_province_datacube_tiled((120.0, 10.0, 120.2, 10.1), 12, 2025)


def test_hls_one_collection_per_search_sensors_merged(mpc):
    dp.set_data_source("hls")
    cube = dp.load_ndvi_stack((120.0, 10.0, 120.05, 10.05), "2025-07-23", "2026-03-20")
    assert mpc.searches == ["hls2-s30", "hls2-l30"]
    assert {b for b, _ in mpc.loads} == {("B04", "B8A", "Fmask"), ("B04", "B05", "Fmask")}
    assert pd.DatetimeIndex(cube.time.values).is_monotonic_increasing
    w0, w1 = mpc.window
    n_s = ((mpc.s2_dates >= w0) & (mpc.s2_dates <= w1)).sum()
    n_l = ((mpc.l30_dates >= w0) & (mpc.l30_dates <= w1)).sum()
    assert cube.sizes["time"] == n_s + n_l


def test_hls_unknown_asset_names_reported():
    from conftest import Item
    with pytest.raises(KeyError, match="nir_x"):
        dp._hls_band_keys(Item("hls2-s30", 9, ["B04", "nir_x", "Fmask"]))
