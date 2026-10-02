"""Shared fixtures: a fake Microsoft Planetary Computer (STAC catalog + odc.stac.load) serving
synthetic rice NDVI, so the tests run without network access, odc-stac or real imagery.

The synthetic field has two crops: an older one peaking 2025-10-15 (harvested ~mid-November) and a
current one planted 2026-01-25 that is still rising in March 2026."""
import sys
import time
import types

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cropgrowth_agent import data_processing as dp

PEAK1 = pd.Timestamp("2025-10-15")
PLANT2 = pd.Timestamp("2026-01-25")


def ndvi_curve(t):
    v = 0.15 + 0.7 * np.exp(-0.5 * ((t - PEAK1).days / 25.0) ** 2)
    r = (t - PLANT2).days
    return v + (0.6 / (1 + np.exp(-(r - 45) / 9.0)) if r > 0 else 0)


class Asset:
    def __init__(self, href):
        self.href = href


class Item:
    def __init__(self, collection, i, bands, bbox=(119.5, 9.5, 121.0, 11.0)):
        self.collection_id, self.id, self.bbox = collection, f"{collection}_{i}", bbox
        self.assets = {b: Asset(f"https://example/{self.id}.{b}.tif") for b in bands}

    @property
    def geometry(self):
        w, s, e, n = self.bbox
        return {"type": "Polygon", "coordinates": [[(w, s), (e, s), (e, n), (w, n), (w, s)]]}


class FakeMPC:
    """STAC catalog + odc.stac.load stand-in. Records every search and load."""

    def __init__(self):
        self.pools = {
            "sentinel-2-l2a": [Item("sentinel-2-l2a", 0, ["B04", "B08", "SCL"])],
            "hls2-s30": [Item("hls2-s30", 0, ["B04", "B8A", "Fmask"])],
            "hls2-l30": [Item("hls2-l30", 0, ["B04", "B05", "Fmask"])],
            "esa-worldcover": [Item("esa-worldcover", 0, ["map"], bbox=(119.0, 9.0, 121.0, 12.0))],
        }
        self.crop_box = None                   # (w, s, e, n) cropland; None = everywhere
        self.bad_ids = set()                   # item ids whose COGs have no CRS
        self.cloudy_every = 0                  # every n-th scene fully cloudy (0 = none)
        self.searches, self.loads = [], []
        self.window = None
        self.s2_dates = pd.date_range("2025-06-01", "2026-12-31", freq="5D")
        self.l30_dates = pd.date_range("2025-06-03", "2026-12-31", freq="8D")

    # -- STAC
    def search(self, collections, bbox, datetime=None, **kw):
        if len(collections) != 1:
            raise RuntimeError("Request must unambiguously select exactly one collection")
        self.searches.append(collections[0])
        if datetime:
            a, b = datetime.split("/")
            self.window = (pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1))
        pool = self.pools.get(collections[0], [])
        hits = [i for i in pool if dp.items_in_bbox([i], bbox)]

        class Result:
            def items(_):
                return list(hits)
        return Result()

    # -- odc.stac.load
    def load(self, items, bands, bbox, crs, resolution, resampling=None, **kw):
        w, s, e, n = bbox
        xs = np.arange(w + resolution / 2, e, resolution)
        ys = np.arange(n - resolution / 2, s, -resolution)
        dims = ("time", "latitude", "longitude")

        def ds(data, times):
            return xr.Dataset({k: (dims, v) for k, v in data.items()},
                              coords={"time": times, "latitude": ys, "longitude": xs})
        if list(bands) == ["map"]:
            X, Y = np.meshgrid(xs, ys)
            crop = np.ones(X.shape, bool) if self.crop_box is None else (
                (X >= self.crop_box[0]) & (X < self.crop_box[2]) & (Y >= self.crop_box[1]) & (Y < self.crop_box[3]))
            return ds({"map": np.where(crop, 40, 10)[None].astype("uint8")}, [pd.Timestamp("2021-01-01")])
        if any(it.id in self.bad_ids for it in items):
            raise AssertionError()             # odc: assert src.crs is not None
        self.loads.append((tuple(bands), tuple(round(v, 4) for v in bbox)))
        dates = self.l30_dates if "B05" in bands else self.s2_dates
        if self.window:
            dates = dates[(dates >= self.window[0]) & (dates <= self.window[1])]
        v = np.array([ndvi_curve(t) for t in dates])[:, None, None] * np.ones((1, ys.size, xs.size))
        if "SCL" in bands:                     # Sentinel-2 L2A DN with the 2022+ BOA offset
            red = np.full(v.shape, 1500, "uint16")
            nir = (1000 + 500 * (1 + v) / (1 - v)).astype("uint16")
            qa = np.full(v.shape, 4, "uint8")
            if self.cloudy_every:
                qa[::self.cloudy_every] = 9
        else:                                  # HLS reflectance x 10000 + Fmask
            red = np.full(v.shape, 500, "int16")
            nir = (500 * (1 + v) / (1 - v)).astype("int16")
            qa = np.zeros(v.shape, "uint8")
            if self.cloudy_every:
                qa[::self.cloudy_every] = 1 << 1
        return ds({bands[0]: red, bands[1]: nir, bands[2]: qa}, dates)


@pytest.fixture
def mpc(monkeypatch):
    m = FakeMPC()
    odc = types.ModuleType("odc")
    stac = types.ModuleType("odc.stac")
    stac.load = m.load
    odc.stac = stac
    monkeypatch.setitem(sys.modules, "odc", odc)
    monkeypatch.setitem(sys.modules, "odc.stac", stac)
    monkeypatch.setattr(dp, "_open_catalog", lambda retries=4, fresh=False: m)
    monkeypatch.setattr(dp, "_asset_is_bad", lambda href: any(b in href for b in m.bad_ids))
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # module state the code under test changes: restored after each test
    for name in ("GRID_SCALE_DEG", "RESOLUTION_M", "DATA_SOURCE", "INTERVAL_DAYS",
                 "COMPOSITE_METHOD", "HLS_MASK_HIGH_AEROSOL", "MAX_SCENE_CLOUD"):
        monkeypatch.setattr(dp, name, getattr(dp, name))
    monkeypatch.setattr(dp, "_BAD_HREFS", set())
    dp.set_resolution(0.005 * dp.M_PER_DEG)    # coarse grid: fast tests
    return m


@pytest.fixture
def boundaries(tmp_path):
    """Province file (3 provinces, 2 regions) and a municipal file, as GeoPackages."""
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import box
    prov = gpd.GeoDataFrame(
        {"Pro_Name": ["Alpha", "Beta", "Palawan"], "Reg_Name": ["R1", "R1", "R4"], "Semester_1": [12, 12, 12],
         "Semester_2": ["6", None, "May"]},          # text, missing and month-name values, as in real tables
        geometry=[box(120.0, 10.0, 120.1, 10.1), box(120.1, 10.0, 120.2, 10.1), box(120.2, 10.0, 120.3, 10.1)],
        crs=4326)
    mun = gpd.GeoDataFrame({"Mun_Name": ["Muñoz Town"]}, geometry=[box(120.0, 10.0, 120.05, 10.05)], crs=4326)
    prov.to_file(tmp_path / "prov.gpkg")
    mun.to_file(tmp_path / "mun.gpkg")
    return {"provinces": str(tmp_path / "prov.gpkg"), "aoi": str(tmp_path / "mun.gpkg")}
