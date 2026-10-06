"""Disk-backed province products and COG mosaics."""
import os

import numpy as np
import pandas as pd
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from cropgrowth_agent import data_processing as dp
from cropgrowth_agent import phenology as ph
from cropgrowth_agent import pipeline as pl
from cropgrowth_agent.mosaic import GridWriter, grid_for_bbox, mosaic_cogs
from cropgrowth_agent.runner import RunConfig, Runner


def _tif(path, x0, y1, arr, res=0.01, nodata=-1):
    arr = np.asarray(arr, "int16")
    arr = arr[None] if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype="int16", crs="EPSG:4326", transform=from_origin(x0, y1, res, res), nodata=nodata) as d:
        d.write(arr)
    return str(path)


def test_grid_is_global():
    assert grid_for_bbox((120.004, 10.0, 120.0505, 10.03), 0.01) == (120.0, 10.03, 6, 3)
    a, b = grid_for_bbox((120.0, 10.0, 121.0, 11.0), 0.01), grid_for_bbox((120.37, 10.2, 120.9, 10.8), 0.01)
    assert round((b[0] - a[0]) / 0.01, 6) == round((b[0] - a[0]) / 0.01)    # same pixel edges


def test_mosaic_misaligned_overlapping_inputs(tmp_path):
    a = np.full((50, 60), 2); a[:, :5] = -1
    b = np.full((40, 40), 3)
    pa = _tif(tmp_path / "a.tif", 120.0, 11.0, a)
    pb = _tif(tmp_path / "b.tif", 120.553, 10.803, b)            # 0.3 px off the grid, overlaps a
    out = mosaic_cogs([pa, pb], str(tmp_path / "m.tif"), log=None)
    with rasterio.open(out["path"]) as m:
        x = m.read(1)
        assert m.nodata == -1 and m.crs.to_epsg() == 4326
    assert (x == 2).sum() == (a == 2).sum()                    # first input wins where they overlap
    assert 0 < (x == 3).sum() < b.size


def test_mosaic_output_is_a_cog_with_overviews(tmp_path):
    big = np.where(np.add.outer(np.arange(1500), np.arange(1500)) % 7 == 0, 3, 2)
    p = _tif(tmp_path / "big.tif", 120.0, 11.0, big, res=0.0005)
    out = mosaic_cogs([p], str(tmp_path / "m.tif"), log=None)
    with rasterio.open(out["path"]) as m:
        assert m.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"
        assert m.profile["tiled"] and m.profile["blockxsize"] == 512
        assert m.overviews(1)                                  # internal overviews
        assert m.compression is not None
        assert np.array_equal(m.read(1), big)


def test_writer_multiband_first_valid_wins(tmp_path):
    w = GridWriter((120.0, 10.0, 120.05, 10.05), 0.01, count=2, nodata=-32768)
    ys, xs = 10.045 - 0.01 * np.arange(5), 120.005 + 0.01 * np.arange(5)
    first = np.full((2, 5, 5), -32768); first[1] = 7            # band 1 empty, band 2 has data
    w.write(first, ys, xs)
    w.write(np.full((2, 5, 5), 9), ys, xs)                     # must not overwrite
    w.to_cog(str(tmp_path / "q.tif"))
    with rasterio.open(tmp_path / "q.tif") as q:
        assert (q.read(2) == 7).all() and (q.read(1) == -32768).all()


def test_tile_sink_keeps_no_tiles(mpc):
    dp.set_data_source("s2")
    seen = []
    out = dp.build_province_datacube_tiled((120.0, 10.0, 120.3, 10.1), 12, 2025, per_tile_fn=ph.run_phenology,
                                           tile_sink=lambda t, tb: seen.append(tb))
    assert isinstance(out, dict) and out["tiles_with_data"] == 3 == len(seen)


@pytest.fixture
def runner(mpc, boundaries, tmp_path):
    cfg = RunConfig(data_source="hls", resolution_m=dp.GRID_SCALE_DEG * dp.M_PER_DEG, tile_deg=0.1,
                    tile_workers=1, date_median_radius=0, vector_path=boundaries["provinces"],
                    region_col="Reg_Name", output_target="local", season_year=2026,
                    local_root=str(tmp_path / "out"), save_phenology_bands=True)
    return Runner(cfg, log=lambda *a: None)


def test_streamed_province_matches_in_memory(runner):
    res = runner.run_periodic("provincial", ["Alpha"], "2026-04-01")
    m = res["success"][0]["maps"][0]
    units = runner.select("provincial", ["Alpha"])
    pheno = runner.process_province(units, "ALPHA", pd.Timestamp("2026-04-01"))
    p = pl.period_of("2026-03-01")
    mem = ph.classify_stage_period(pheno, p.start, p.end)
    with rasterio.open(m["uri"]) as src:
        disk = src.read(1)
        t = src.transform
        xs = t.c + (np.arange(src.width) + 0.5) * t.a
        ys = t.f + (np.arange(src.height) + 0.5) * t.e
    ref = mem.reindex(y=ys, x=xs, method="nearest", tolerance=dp.GRID_SCALE_DEG / 2).fillna(-1).values
    assert (disk != -1).sum() > 0
    assert np.array_equal(disk, ref)
    qa = runner.list_outputs("periodic", "phenology_dates")["files"]
    with rasterio.open(os.path.join(runner.store.root, qa[0])) as q:
        assert q.count == len(ph.QA_BANDS) and q.descriptions[0] == "plant"


def test_national_run_builds_mosaic(runner):
    res = runner.run_periodic("national", None, "2026-04-01")
    assert len(res["success"]) == 3                             # Palawan included (no default exclusion)
    mz = res["mosaic"]["mosaics"][0]
    assert mz["period"] == "202603" and mz["provinces"] == 3
    assert os.path.basename(mz["uri"]) == "dry2026_PHILIPPINES_202603.tiff"
    total = {}
    for s in res["success"]:
        for k, v in s["maps"][0]["hectares"].items():
            total[k] = total.get(k, 0) + v
    for k, v in mz["hectares"].items():
        assert v == pytest.approx(total[k], rel=0.01)
    files = runner.list_outputs("periodic", "mosaic")["files"]
    assert any(f.endswith("_PHILIPPINES_202603_summary.csv") for f in files)
    again = runner.mosaic_periodic("regional", ["R1"], "2026-04-01")
    assert again["scope"] == "R1" and again["mosaics"][0]["provinces"] == 2
