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


# ---- stages are not forced when the series doesn't show them (cases from real wet-season pixels:
#      10-day composites May-Dec 2026, clear observations end 2026-09-28)
_T = pd.date_range("2026-05-01", "2026-12-31", freq="10D")


def _pixels(**series):
    last = pd.Timestamp("2026-09-28")
    cols = []
    for pts in series.values():
        d = pd.to_datetime([p[0] for p in pts])
        x = np.interp(_T.values.astype("int64"), d.values.astype("int64"), [p[1] for p in pts])
        x[_T > last] = np.nan
        cols.append(x)
    X = np.stack(cols, axis=1)[:, None, :]
    ds = xr.DataArray(X, dims=("time", "y", "x"),
                      coords={"time": _T, "y": [10.0], "x": np.arange(X.shape[2])}).to_dataset(name="NDVI")
    ds.attrs.update(spatial_dims=["y", "x"], window_start="2026-05-01", window_end="2026-12-31")
    out = ph.run_phenology(ds)
    return {k: out.isel(y=0, x=i) for i, k in enumerate(series)}, out


def test_no_peak_or_harvest_while_ndvi_is_still_high():
    p, out = _pixels(high_end=[("2026-05-01", .37), ("2026-05-20", .59), ("2026-06-20", .65), ("2026-07-10", .37),
                               ("2026-08-01", .50), ("2026-09-01", .72), ("2026-09-20", .81), ("2026-09-28", .77)])
    q = p["high_end"]
    assert int(q["qc"]) == ph.QC_ONGOING_PRE
    assert all(np.isnan(float(q[b])) for b in ("heading", "peak", "maturity", "harvest"))
    assert np.isfinite(float(q["emergence"]))
    assert ph.classify_stage(out, "2026-10-01").values[0, 0] == 3          # reproductive, not harvested


def test_full_cycle_keeps_its_dates():
    p, _ = _pixels(full=[("2026-05-01", .23), ("2026-05-20", .22), ("2026-06-01", .24), ("2026-07-10", .72),
                         ("2026-07-31", .85), ("2026-08-31", .77), ("2026-09-20", .71), ("2026-09-28", .44)])
    q = p["full"]
    assert int(q["qc"]) == ph.QC_COMPLETE
    for b in ("plant", "emergence", "panicle_init", "heading", "peak", "maturity", "harvest"):
        assert np.isfinite(float(q[b])), b
    order = [float(q[b]) for b in ("plant", "emergence", "panicle_init", "heading", "peak", "maturity", "harvest")]
    assert order == sorted(order)


def test_planting_not_reported_when_rise_starts_at_first_image():
    p, _ = _pixels(early=[("2026-05-01", .21), ("2026-05-10", .27), ("2026-06-01", .55), ("2026-06-21", .73),
                          ("2026-07-21", .67), ("2026-09-20", .54), ("2026-09-28", .46)])
    q = p["early"]
    assert np.isnan(float(q["plant"]))                                     # trough not observed
    assert np.isfinite(float(q["emergence"])) and np.isfinite(float(q["peak"]))


def test_reported_dates_are_never_out_of_order():
    rng = np.random.default_rng(0)
    base = 0.2 + 0.6 * np.exp(-0.5 * (((_T - pd.Timestamp("2026-07-25")).days.values) / 22.0) ** 2)
    X = base[:, None] + rng.normal(0, 0.06, (len(_T), 400))
    X[rng.random(X.shape) < 0.3] = np.nan
    ds = xr.DataArray(X[:, None, :].astype("float32"), dims=("time", "y", "x"),
                      coords={"time": _T, "y": [10.0], "x": np.arange(400)}).to_dataset(name="NDVI")
    ds.attrs.update(spatial_dims=["y", "x"], window_start="2026-05-01")
    out = ph.run_phenology(ds)
    D = np.stack([out[b].values[0] for b in ph.DATE_BANDS])
    for i in range(D.shape[1]):
        v = D[:, i][np.isfinite(D[:, i])]
        assert (np.diff(v) >= 0).all(), dict(zip(ph.DATE_BANDS, D[:, i]))
