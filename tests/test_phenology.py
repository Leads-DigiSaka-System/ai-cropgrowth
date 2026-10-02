import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cropgrowth_agent import phenology as ph

from conftest import ndvi_curve


def _cube(dates, values, n=3):
    v = np.asarray(values, np.float32)[:, None, None] * np.ones((1, n, n), np.float32)
    ds = xr.DataArray(v, dims=("time", "y", "x"),
                      coords={"time": dates, "y": np.arange(n)[::-1] * 0.01 + 10, "x": np.arange(n) * 0.01 + 120}
                      ).to_dataset(name="NDVI")
    ds.attrs.update(spatial_dims=["y", "x"], window_start=str(dates[0].date()), window_end=str(dates[-1].date()))
    return ds


def _season():
    dates = pd.date_range("2025-11-01", "2026-06-30", freq="10D")
    d = (dates - pd.Timestamp("2026-02-15")).days.values
    return _cube(dates, 0.15 + 0.7 * np.exp(-0.5 * (d / 25.0) ** 2))


def test_single_cycle_dates_in_order():
    out = ph.run_phenology(_season())
    p = {b: float(out[b][0, 0]) for b in ph.DATE_BANDS}
    assert float(out["qc"][0, 0]) == ph.QC_COMPLETE
    order = [p[b] for b in ph.DATE_BANDS]
    assert order == sorted(order)
    assert abs(p["peak"] - ph.to_epoch_days("2026-02-15")[0]) < 6
    assert 50 <= float(out["season_length"][0, 0]) <= 180


def test_last_cycle_peak_follows_young_crop():
    dates = pd.date_range("2025-07-23", "2026-03-20", freq="10D")
    ds = _cube(dates, [ndvi_curve(t) for t in dates])
    common = dict(PEAK_SEARCH_DAYS=(30, 240))
    last = ph.run_phenology(ds, PEAK_SELECT="last", **common)
    best = ph.run_phenology(ds, PEAK_SELECT="max", **common)
    assert ph.classify_stage(last, "2026-03-20").values[0, 0] == 2      # vegetative
    assert ph.classify_stage(best, "2026-03-20").values[0, 0] == 5      # old crop, harvested


def test_smooth_series_matches_pipeline():
    x = _season()["NDVI"].values[:, 0, 0].copy()
    x[[3, 7]] = np.nan
    s, _ = ph.smooth_series(x)
    _, s_full = ph._clean_and_smooth(x[:, None], dict(ph.DEFAULTS), np.array([True]))
    assert np.array_equal(s, s_full[:, 0])


def test_month_classifier_is_period_classifier():
    out = ph.run_phenology(_season())
    for m in ["2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05"]:
        m0 = pd.Timestamp(m + "-01")
        a = ph.classify_stage_month(out, m).values
        b = ph.classify_stage_period(out, m0, m0 + pd.offsets.MonthBegin(1)).values
        assert np.array_equal(a, b)
        assert np.array_equal(ph.classify_stage_month(out, m, "midmonth").values,
                              ph.classify_stage(out, m0 + pd.Timedelta(days=14)).values)


def test_recent_stage_reports_data_age():
    ds = _season()
    out = ph.run_phenology(ds)
    rec = ph.recent_stage(out, "2026-07-10")
    last = pd.Timestamp(ds.time.values[-1])
    assert float(rec["data_age_days"][0, 0]) == (pd.Timestamp("2026-07-10") - last).days
    assert rec.attrs["as_of"] == "2026-07-10"


def test_cog_writers_roundtrip(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    out = ph.run_phenology(_season())
    st = ph.classify_stage_month(out, "2026-02")
    ph.write_stage_cog(st, str(tmp_path / "s.tif"))
    rec = ph.recent_stage(out, "2026-03-01")
    ph.write_recent_cog(rec, str(tmp_path / "r.tif"))
    with rasterio.open(tmp_path / "s.tif") as r:
        assert np.array_equal(r.read(1), st.values)
    with rasterio.open(tmp_path / "r.tif") as r:
        assert r.descriptions == ("growth_stage", "data_age_days", "qc")
