"""
phenology.py
==================================================================
Formula-based rice growth-stage extraction from a Sentinel-2 NDVI datacube.
Takes the place of inference.py (CNN-LSTM): no model, no training — only
thresholds, Savitzky-Golay smoothing and derivatives of the per-pixel NDVI
time series.

Consumes the datacube produced by data_processing.build_province_datacube:
    ds['NDVI']  dims (time, y, x), float32, NaN = cloud / gap / non-crop
    ds.attrs['spatial_dims'] = ['y', 'x'], ds.attrs['window_start']

Method (per pixel, one rice cycle per season window)
----------------------------------------------------
  NDVI(t) --(SCL mask, despike, linear gap-fill)--> SG smoothing:
        NDVI_s(t) = SG(NDVI(t))
        G(t)  = dNDVI_s/dt           G'(t) = d2NDVI_s/dt2

  D_peak      = argmax_t NDVI_s(t)                  (inside PEAK_SEARCH_DAYS)
  NDVI_max    = NDVI_s(D_peak)
  NDVI_min    = trough before the peak (rising limb) and after the peak
                (falling limb)                        (BASE_MODE='separate')
                or one min of both troughs            (BASE_MODE='single')
  A           = NDVI_max - NDVI_min
  NDVI_15     = NDVI_min + 0.15 * A          (<=> NDVI_norm = 0.15)

  Growth stage (transition date)       rule
  -----------------------------------  ---------------------------------------
  Planting / transplanting   D_plant   NDVI_s minimum immediately before the
                                       rising NDVI_15 crossing
  Emergence / establishment  D_emerg   rising crossing of NDVI_15
                                       = min{t : NDVI_s(t) >= NDVI_15}
  Tillering / vegetative     D_till    onset of rapid increase: argmax G'(t)
                                       in [D_emerg, D_PI]  (TILLER_RULE='max_accel')
                                       or NDVI_norm = TILLER_NORM crossing ('norm')
  Panicle initiation         D_PI      inflection of the rising limb:
                                       argmax G(t) in [D_emerg, D_peak]
  Heading / flowering        D_head    NDVI approaches NDVI_max: rising
                                       crossing of NDVI_norm = HEADING_NORM
  (peak)                     D_peak    NDVI_s reaches NDVI_max
  Grain filling              D_peak .. D_mat  gradual decline after the peak
  Maturity                   D_mat     steepest decline: argmin G(t) after peak
  Harvest                    D_harv    falling crossing of NDVI_15
                                       = min{t > D_peak : NDVI_s(t) <= NDVI_15}

  Crossings and extrema are refined to sub-composite precision (linear
  interpolation for crossings, 3-point parabola for extrema), so dates are
  not quantised to the INTERVAL_DAYS grid.

Intermediate outputs of run_phenology (all (y, x), float32, NaN = nodata) —
the transition dates the stage maps are computed from (optionally saved as a
multi-band COG for QA; not the main product):
  plant, emergence, tillering, panicle_init, heading, peak, maturity, harvest
        -> dates as days since 1970-01-01 (converted to DOY at export)
  season_length   harvest - emergence (days; the two 15 % crossings)
  ndvi_min, ndvi_max, ndvi15, n_obs, max_gap_days, qc

MAIN PRODUCT: one single-band stage map per month — classify_stage_month(ds,
'YYYY-MM') gives each pixel the stage it spends most of that month in
(classify_stage(ds, date) gives the stage on one exact date). 6 classes:
  0 no rice cycle detected
  1 pre-planting / transplanting   (before emergence)
  2 vegetative                     (emergence -> panicle initiation)
  3 reproductive                   (panicle initiation -> peak NDVI / flowering)
  4 ripening / maturity            (peak NDVI -> harvest)
  5 harvested / post-harvest       (after harvest)
==================================================================
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter

from .data_processing import NDVI_BAND

# ------------------------------------------------------------------
# Config (all overridable per call via run_phenology(..., **overrides))
# ------------------------------------------------------------------
DEFAULTS = dict(
    THRESHOLD_FRAC      = 0.15,       # NDVI_15 = min + 0.15 A
    HEADING_NORM        = 0.90,       # NDVI_norm at which "approaching max" starts
    TILLER_RULE         = "max_accel",# tillering onset: 'max_accel' = argmax G'(t) (derivative rule;
                                      #   on 10-d composites it usually lands within a few days
                                      #   of emergence) or 'norm' = rising crossing of TILLER_NORM
    TILLER_NORM         = 0.30,       # used when TILLER_RULE='norm' (calibrate with field data)
    BASE_MODE           = "separate", # 'separate': NDVI_15 formula applied per limb — rising
                                      #   limb uses the pre-planting trough, falling limb the
                                      #   post-harvest trough (stubble/ratoon keeps the latter
                                      #   higher; with one min, harvest is often never crossed)
                                      # 'single': one NDVI_min = min of both troughs
    PLANT_FROM          = "smoothed", # trough for planting date: 'smoothed' (NDVI_s) or
                                      # 'raw' (despiked, gap-filled NDVI; sharper flood minimum)
    SG_WINDOW           = 5,          # SG window in composites (5 x 10 d = 50 d)
    SG_POLYORDER        = 2,
    PEAK_SEARCH_DAYS    = (45, 210),  # peak allowed this many days after window start
    PEAK_SELECT         = "max",      # 'max': highest NDVI_s in the search range (one
                                      #   season per window — season products)
                                      # 'last': most recent cycle — the latest local max
                                      #   (or a still-rising end) that rises >= MIN_AMPLITUDE
                                      #   above its pre-peak trough; for rolling windows
                                      #   that can hold two crops (recent products)
    PRE_PEAK_MAX_DAYS   = 110,        # trough search: at most this far before the peak
    POST_PEAK_MAX_DAYS  = 90,         # trough search: at most this far after the peak
    MIN_OBS             = 6,          # min clear composites in the window
    MIN_AMPLITUDE       = 0.20,       # rice-likeness QC
    MIN_PEAK_NDVI       = 0.50,
    MAX_BASE_NDVI       = 0.45,       # perennial / tree cover never drops this low
    SEASON_LEN_DAYS     = (50, 180),  # emergence -> harvest plausible range
    PEAK_CONFIRM_FRAC   = 0.10,       # a peak counts only once NDVI_s has fallen this fraction of the
                                      #   amplitude after it (in observed data); otherwise the crop is
                                      #   still at / before its peak (QC 2) — no peak, maturity, harvest
    MIN_FALL_FRAC       = 0.40,       # BASE_MODE='separate': the post-peak trough is used as the falling
                                      #   base only if NDVI fell at least this fraction of the amplitude;
                                      #   a shallow "trough" (e.g. just the last observation) is not a
                                      #   harvest, so the rising base is used instead
    DESPIKE_DROP        = 0.15,       # drop single dips this far below neighbour mean ...
    DESPIKE_MIN_NEIGHBOR= 0.40,       # ... only when BOTH neighbours are vegetated
                                      #     (protects the genuine flooded-paddy trough)
)

BATCH_PIXELS = 400_000                # pixels per vectorised batch (memory bound)
DATE_BANDS = ["plant", "emergence", "tillering", "panicle_init",
              "heading", "peak", "maturity", "harvest"]

# QC codes
QC_COMPLETE        = 0    # full cycle, harvest detected
QC_ONGOING_POST    = 1    # past peak, harvest crossing not yet observed
QC_ONGOING_PRE     = 2    # still rising at last observation (provisional max)
QC_NO_DATA         = 10   # < MIN_OBS clear composites
QC_NO_PEAK         = 11   # no interior peak in PEAK_SEARCH_DAYS
QC_LOW_AMPLITUDE   = 12
QC_LOW_PEAK        = 13
QC_HIGH_BASE       = 14
QC_NO_RISE         = 15   # no rising 15 % crossing before the peak
QC_BAD_LENGTH      = 16   # season length outside SEASON_LEN_DAYS
QC_DESCRIPTION = {
    0: "complete cycle", 1: "ongoing, past peak", 2: "ongoing, before / at peak (peak not confirmed)",
    10: "too few clear observations", 11: "no interior peak",
    12: "amplitude too small", 13: "peak NDVI too low", 14: "base NDVI too high",
    15: "no rising 15% crossing", 16: "season length implausible",
}

# Stage classes for classify_stage
STAGE_NODATA = -1
STAGE_CLASSES = {
    0: "no rice cycle detected",
    1: "pre-planting / transplanting",
    2: "vegetative",
    3: "reproductive",
    4: "ripening / maturity",
    5: "harvested / post-harvest",
}
# Class boundaries (transition-date band that STARTS each class):
#   1 pre-planting / transplanting : before emergence (fallow, land prep, flooded
#                                    paddy, transplanting — NDVI below NDVI_15)
#   2 vegetative                   : emergence (rising NDVI_15)  -> panicle initiation
#                                    (establishment + tillering)
#   3 reproductive                 : panicle initiation (max G)  -> peak NDVI
#                                    (booting, heading, flowering ~ NDVI_max)
#   4 ripening / maturity          : peak NDVI                   -> harvest
#                                    (grain filling + maturity, NDVI declining)
#   5 harvested / post-harvest     : harvest (falling NDVI_15)
STAGE_START_BAND = [(2, "emergence"), (3, "panicle_init"), (4, "peak"), (5, "harvest")]

EPOCH = pd.Timestamp("1970-01-01")


def to_epoch_days(dates):
    """datetime-like -> float days since 1970-01-01."""
    d = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(dates)))
    return ((d - EPOCH) / pd.Timedelta(days=1)).values.astype(np.float64)


def from_epoch_days(days):
    """float days since 1970-01-01 -> DatetimeIndex (always 1-D)."""
    return EPOCH + pd.to_timedelta(np.atleast_1d(np.asarray(days, dtype=float)), unit="D")


# ==================================================================
# Vectorised helpers  (arrays are (T, N): time x pixels)
# ==================================================================
def _prev_next_valid(valid):
    T, N = valid.shape
    idx = np.arange(T)[:, None]
    prev = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(valid, idx, T)[::-1], axis=0)[::-1]
    return prev, nxt


def gap_fill_linear(X):
    """Linear interpolation over NaN along time; nearest-value extension at
    the ends. All-NaN columns stay NaN."""
    X = np.asarray(X, dtype=np.float32)
    T, N = X.shape
    valid = np.isfinite(X)
    prev, nxt = _prev_next_valid(valid)
    cols = np.arange(N)[None, :]
    has_p, has_n = prev >= 0, nxt < T
    xp = np.where(has_p, X[np.clip(prev, 0, T - 1), cols], np.nan)
    xn = np.where(has_n, X[np.clip(nxt, 0, T - 1), cols], np.nan)
    idx = np.arange(T)[:, None].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (idx - prev) / np.maximum(nxt - prev, 1)
        lin = xp + w * (xn - xp)
    out = np.where(valid, X,
          np.where(has_p & has_n, lin, np.where(has_p, xp, xn)))
    return out.astype(np.float32)


def _despike(X, drop, min_neighbor):
    """Remove single-composite dips (residual cloud/haze) inside a vegetated
    stretch. Returns X with spikes set to NaN."""
    F = gap_fill_linear(X)
    prv = np.vstack([F[:1], F[:-1]])
    nxt = np.vstack([F[1:], F[-1:]])
    spike = (np.isfinite(X)
             & (X < 0.5 * (prv + nxt) - drop)
             & (np.minimum(prv, nxt) > min_neighbor))
    spike[0] = spike[-1] = False
    out = X.copy()
    out[spike] = np.nan
    return out


def _range_mask(T, lo, hi):
    idx = np.arange(T)[:, None]
    return (idx >= lo[None, :]) & (idx <= hi[None, :])


def _masked_argmax(A, mask):
    return np.argmax(np.where(mask, A, -np.inf), axis=0)


def _masked_argmin(A, mask):
    return np.argmin(np.where(mask, A, np.inf), axis=0)


def _take(A, idx):
    return np.take_along_axis(A, np.clip(idx, 0, A.shape[0] - 1)[None, :], axis=0)[0]


def _parabolic_offset(A, idx):
    """Sub-step offset of an extremum at idx from a 3-point parabola."""
    T = A.shape[0]
    inner = (idx > 0) & (idx < T - 1)
    y0, y1, y2 = _take(A, idx - 1), _take(A, idx), _take(A, idx + 1)
    den = y0 - 2 * y1 + y2
    with np.errstate(invalid="ignore", divide="ignore"):
        off = 0.5 * (y0 - y2) / den
    off = np.where(inner & np.isfinite(off) & (den != 0), off, 0.0)
    return np.clip(off, -0.5, 0.5)


def _crossing_time(S, a_idx, thr, t, dt):
    """Linear-interpolated time where S crosses thr between a_idx and a_idx+1."""
    sa, sb = _take(S, a_idx), _take(S, a_idx + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (thr - sa) / (sb - sa)
    frac = np.clip(np.where(np.isfinite(frac), frac, 0.5), 0.0, 1.0)
    return t[np.clip(a_idx, 0, len(t) - 1)] + frac * dt


def _last_cycle_peak(S, ip_max, p_lo, p_hi, first_obs, last_obs, pre_steps, min_amp):
    """Index of the most recent cycle's peak per pixel: the latest local
    maximum in [p_lo, p_hi] (or the last observation if NDVI is still rising
    there) whose rise above the minimum of the preceding pre_steps composites
    is >= min_amp. Falls back to ip_max where no candidate qualifies."""
    T, N = S.shape
    idx = np.arange(T)[:, None]
    prev = np.vstack([np.full((1, N), -np.inf), S[:-1]])
    nxt = np.vstack([S[1:], np.full((1, N), np.inf)])
    local_max = (S >= prev) & (S > nxt)
    rising_end = (idx == last_obs[None, :]) & (S > prev)
    cand = (local_max | rising_end) & (idx >= p_lo[None, :]) & (idx <= p_hi[None, :])
    # minimum over the pre_steps composites before each index (not before first_obs)
    left_min = np.full_like(S, np.inf)
    for k in range(1, pre_steps + 1):
        sh = np.vstack([np.full((k, N), np.inf), S[:-k]])
        sh = np.where(idx - k >= first_obs[None, :], sh, np.inf)
        left_min = np.minimum(left_min, sh)
    cand &= (S - left_min) >= min_amp
    has = cand.any(0)
    last = T - 1 - np.argmax(cand[::-1], axis=0)
    return np.where(has, last, ip_max)


# ==================================================================
# Core: phenology for a batch of pixel time series
# ==================================================================
def extract_phenology(X, t_days, window_start_days, cfg=None):
    """
    X        : (T, N) NDVI composites, NaN = missing
    t_days   : (T,) composite dates, days since epoch, regular spacing
    Returns dict of (N,) float arrays (dates in epoch days; NaN = undefined).
    """
    c = dict(DEFAULTS); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float32)
    T, N = X.shape
    t = np.asarray(t_days, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    idx = np.arange(T)[:, None]
    nan = np.full(N, np.nan, np.float32)
    out = {k: nan.copy() for k in DATE_BANDS + [
        "season_length", "ndvi_min", "ndvi_max", "ndvi15", "max_gap_days", "last_obs"]}

    obs = np.isfinite(X)
    n_obs = obs.sum(0)
    out["n_obs"] = n_obs.astype(np.float32)
    qc = np.full(N, QC_NO_DATA, np.int16)
    ok = n_obs >= c["MIN_OBS"]
    if not ok.any():
        out["qc"] = qc.astype(np.float32)
        return out

    # first / last real observation (searches never extrapolate past them)
    first_obs = np.argmax(obs, axis=0)
    last_obs = T - 1 - np.argmax(obs[::-1], axis=0)

    # longest gap between consecutive clear composites (days)
    prev, _ = _prev_next_valid(obs)
    prev_shift = np.vstack([np.full((1, N), -1), prev[:-1]])
    gaps = np.where(obs & (prev_shift >= 0), idx - prev_shift, 0)
    out["max_gap_days"] = np.where(ok, gaps.max(0) * dt, np.nan).astype(np.float32)
    # date of the newest clear composite (epoch days): how fresh a stage is
    out["last_obs"] = np.where(obs.any(0), t[last_obs], np.nan).astype(np.float32)

    # ---- clean + smooth -------------------------------------------------
    F, S = _clean_and_smooth(X, c, ok)
    G = np.gradient(S, dt, axis=0)                          # dNDVI_s/dt  (per day)
    G2 = np.gradient(G, dt, axis=0)                         # d2NDVI_s/dt2

    # ---- peak -----------------------------------------------------------
    rel = (t - window_start_days)
    p_lo_d, p_hi_d = c["PEAK_SEARCH_DAYS"]
    p_lo = np.full(N, int(np.searchsorted(rel, p_lo_d)))
    p_hi = np.full(N, int(np.searchsorted(rel, p_hi_d, side="right")) - 1)
    p_lo = np.maximum(p_lo, first_obs)
    p_hi = np.minimum(p_hi, last_obs)
    has_range = p_hi >= p_lo
    ip = _masked_argmax(S, _range_mask(T, p_lo, p_hi))
    if c["PEAK_SELECT"] == "last":
        ip = _last_cycle_peak(S, ip, p_lo, p_hi, first_obs, last_obs,
                              int(round(c["PRE_PEAK_MAX_DAYS"] / dt)), c["MIN_AMPLITUDE"])
    s_ip = _take(S, ip)
    left_ok = (ip - 1 >= first_obs) & (_take(S, ip - 1) <= s_ip)
    right_ok = (ip + 1 <= last_obs) & (_take(S, ip + 1) <= s_ip)
    pre_peak = has_range & left_ok & (ip == last_obs)       # still rising at the end
    is_peak = has_range & left_ok & (right_ok | pre_peak)
    qc = np.where(ok & ~is_peak, QC_NO_PEAK, qc)
    live = ok & is_peak

    # ---- troughs & threshold -------------------------------------------
    pre_steps = int(round(c["PRE_PEAK_MAX_DAYS"] / dt))
    post_steps = int(round(c["POST_PEAK_MAX_DAYS"] / dt))
    l_lo = np.maximum(first_obs, ip - pre_steps)
    l_hi = ip - 1
    it_l = _masked_argmin(S, _range_mask(T, l_lo, l_hi))
    r_lo, r_hi = ip + 1, np.minimum(last_obs, ip + post_steps)
    has_r = r_hi >= r_lo
    it_r = _masked_argmin(S, _range_mask(T, r_lo, r_hi))

    vmax = s_ip
    vmin_l = _take(S, it_l)
    vmin_r = np.where(has_r, _take(S, it_r), np.nan)
    if c["BASE_MODE"] == "separate":
        base_rise = vmin_l
        base_fall = np.where(has_r, vmin_r, vmin_l)
    else:                                                   # single NDVI_min per cycle
        base_rise = np.where(has_r, np.minimum(vmin_l, vmin_r), vmin_l)
        base_fall = base_rise
    amp = vmax - base_rise
    # the decline seen after the peak, in observed data only
    fall_amp = np.where(has_r, vmax - vmin_r, 0.0)
    # a shallow post-peak "trough" (data ends near the top) is not a harvest base
    base_fall = np.where(fall_amp < c["MIN_FALL_FRAC"] * amp, base_rise, base_fall)
    # peak confirmed only once NDVI has visibly turned down after it; otherwise the
    # crop is still at / before its peak and peak, maturity and harvest stay undefined
    pre_peak = pre_peak | (is_peak & (fall_amp < c["PEAK_CONFIRM_FRAC"] * amp))
    thr_rise = base_rise + c["THRESHOLD_FRAC"] * (vmax - base_rise)
    thr_fall = base_fall + c["THRESHOLD_FRAC"] * (vmax - base_fall)
    thr_head = base_rise + c["HEADING_NORM"] * (vmax - base_rise)

    bad = live & (amp < c["MIN_AMPLITUDE"])
    qc = np.where(bad, QC_LOW_AMPLITUDE, qc); live &= ~bad
    bad = live & ~pre_peak & (vmax < c["MIN_PEAK_NDVI"])    # young crops exempt
    qc = np.where(bad, QC_LOW_PEAK, qc); live &= ~bad
    bad = live & (base_rise > c["MAX_BASE_NDVI"])
    qc = np.where(bad, QC_HIGH_BASE, qc); live &= ~bad

    # ---- emergence: rising NDVI_15 crossing (the one just before the peak)
    below = (S < thr_rise[None, :]) & (idx < ip[None, :]) & (idx >= l_lo[None, :])
    has_rise = below.any(0)
    j = T - 1 - np.argmax(below[::-1], axis=0)              # last below-threshold index
    bad = live & ~has_rise
    qc = np.where(bad, QC_NO_RISE, qc); live &= ~bad
    d_emerg = _crossing_time(S, j, thr_rise, t, dt)

    # ---- planting: NDVI minimum immediately before that crossing --------
    P = S if c["PLANT_FROM"] == "smoothed" else F          # SG rounds the sharp flood trough
    i_plant = _masked_argmin(P, _range_mask(T, l_lo, j))
    d_plant = t[i_plant] + _parabolic_offset(P, i_plant) * dt
    # trough at the very first observation: NDVI may have been lower before the data
    # starts, so the planting date is not observed
    plant_ok = i_plant > first_obs

    # ---- peak date --------------------------------------------------------
    d_peak = t[ip] + _parabolic_offset(S, ip) * dt

    # ---- panicle initiation: max growth rate (inflection of rising limb) --
    pi_lo, pi_hi = j, np.where(pre_peak, ip, ip - 1)
    i_pi = _masked_argmax(G, _range_mask(T, pi_lo, np.maximum(pi_hi, pi_lo)))
    d_pi = t[i_pi] + _parabolic_offset(G, i_pi) * dt
    pi_valid = ~pre_peak | (i_pi < ip - 1)                  # growth rate already turned down
    # the maximum growth rate must be inside the rising limb, not on its first or last
    # step: at an edge the true maximum may lie outside the data
    pi_valid &= (i_pi > pi_lo) & (i_pi < pi_hi)

    # ---- tillering: onset of rapid increase = max acceleration ----------
    if c["TILLER_RULE"] == "norm":                          # calibratable level crossing
        thr_til = base_rise + c["TILLER_NORM"] * (vmax - base_rise)
        below_t = (S < thr_til[None, :]) & (idx < ip[None, :]) & (idx >= j[None, :])
        jt = T - 1 - np.argmax(below_t[::-1], axis=0)
        til_ok = below_t.any(0)
        d_til = _crossing_time(S, jt, thr_til, t, dt)
    else:                                                   # max acceleration (G' peak)
        i_til = _masked_argmax(G2, _range_mask(T, j, np.maximum(i_pi, j)))
        d_til = t[i_til] + _parabolic_offset(G2, i_til) * dt
        til_ok = (i_til > j) & (i_til < i_pi) & pi_valid    # interior, as for PI

    # ---- heading: NDVI_norm reaches HEADING_NORM (approaching max) ------
    below_h = (S < thr_head[None, :]) & (idx < ip[None, :]) & (idx >= j[None, :])
    has_h = below_h.any(0)
    jh = T - 1 - np.argmax(below_h[::-1], axis=0)
    d_head = _crossing_time(S, jh, thr_head, t, dt)         # only when the crossing is observed

    # ---- harvest: first falling NDVI_15 crossing after the peak ---------
    fall = (S <= thr_fall[None, :]) & (idx > ip[None, :]) & (idx <= last_obs[None, :])
    has_harv = fall.any(0) & ~pre_peak
    k = np.argmax(fall, axis=0)                             # first index at/below
    d_harv = _crossing_time(S, k - 1, thr_fall, t, dt)

    # ---- maturity: steepest decline between peak and harvest ------------
    m_hi = np.where(has_harv, k, last_obs)
    m_mask = _range_mask(T, ip + 1, m_hi)
    has_m = m_mask.any(0) & ~pre_peak
    i_mat = _masked_argmin(G, m_mask)
    d_mat = t[i_mat] + _parabolic_offset(G, i_mat) * dt
    # only a real maturity onset if the decline has actually steepened then eased
    # (or harvest seen); otherwise leave undefined for an ongoing cycle
    has_m &= has_harv | (i_mat < m_hi)
    has_m &= fall_amp >= c["MIN_FALL_FRAC"] * amp           # a real decline, not a wobble at the top

    # ---- chronological order: a date that contradicts its neighbours is
    # dropped (left undefined), never moved onto them -------------------
    plant_ok &= d_plant <= d_emerg
    til_ok &= d_til >= d_emerg
    pi_valid &= d_pi >= d_emerg
    til_ok &= ~pi_valid | (d_til <= d_pi)
    head_ok = has_h & ~pre_peak & (d_head <= d_peak)
    head_ok &= ~pi_valid | (d_head >= d_pi)
    has_m &= d_mat >= d_peak
    has_m &= ~has_harv | (d_mat <= d_harv)

    season = d_harv - d_emerg
    lo_len, hi_len = c["SEASON_LEN_DAYS"]
    bad = live & has_harv & ((season < lo_len) | (season > hi_len))
    qc = np.where(bad, QC_BAD_LENGTH, qc); live &= ~bad

    qc = np.where(live, np.where(pre_peak, QC_ONGOING_PRE,
                        np.where(has_harv, QC_COMPLETE, QC_ONGOING_POST)), qc)

    def put(name, val, cond=True):
        out[name] = np.where(live & cond, val, np.nan).astype(np.float32)

    put("plant", d_plant, plant_ok)
    put("emergence", d_emerg)
    put("tillering", d_til, til_ok)
    put("panicle_init", d_pi, pi_valid)
    put("heading", d_head, head_ok)
    put("peak", d_peak, ~pre_peak)
    put("maturity", d_mat, has_m)
    put("harvest", d_harv, has_harv)
    put("season_length", season, has_harv)
    put("ndvi_min", base_rise)
    put("ndvi_max", vmax)
    put("ndvi15", thr_rise)
    out["qc"] = np.where(ok | (qc != QC_NO_DATA), qc, QC_NO_DATA).astype(np.float32)
    return out


def _clean_and_smooth(X, c, ok):
    """Despike -> linear gap-fill -> Savitzky-Golay. X (T, N); ok (N,) marks
    pixels with enough observations (others are zeroed so SG stays NaN-free).
    Returns (F gap-filled, S smoothed), both float32 (T, N)."""
    T = X.shape[0]
    F = gap_fill_linear(_despike(X, c["DESPIKE_DROP"], c["DESPIKE_MIN_NEIGHBOR"]))
    F = np.where(ok[None, :], F, 0.0).astype(np.float32)
    win = min(c["SG_WINDOW"], T if T % 2 == 1 else T - 1)
    win = max(win, c["SG_POLYORDER"] + 2 + (c["SG_POLYORDER"] % 2 == 1))
    if win % 2 == 0:
        win += 1
    S = savgol_filter(F, win, c["SG_POLYORDER"], axis=0, mode="interp").astype(np.float32)
    return F, S


def smooth_series(X, cfg=None):
    """Convenience for plotting: the exact cleaned + SG-smoothed curve used
    by extract_phenology (same window rules and MIN_OBS masking).
    X: (T,) or (T, N). Returns (S, F)."""
    c = dict(DEFAULTS); c.update(cfg or {})
    X = np.asarray(X, np.float32)
    one = X.ndim == 1
    X = X[:, None] if one else X
    ok = np.isfinite(X).sum(0) >= c["MIN_OBS"]
    F, S = _clean_and_smooth(X, c, ok)
    return (S[:, 0], F[:, 0]) if one else (S, F)


# ==================================================================
# Pixelwise run over a datacube  (replaces run_cnnlstm_inference)
# ==================================================================
def run_phenology(ds, batch_size=None, **overrides):
    """
    ds : Dataset with NDVI (time, y, x) + attrs window_start.
    Returns a Dataset of (y, x) float32 products (see module docstring).
    Non-crop / masked pixels (all-NaN series) come out NaN everywhere
    (qc = NaN), so they stay nodata rather than "no rice".
    """
    batch_size = batch_size or BATCH_PIXELS
    da = ds[NDVI_BAND]
    y_dim, x_dim = ds.attrs.get("spatial_dims") or [d for d in da.dims if d != "time"]
    arr = da.transpose("time", y_dim, x_dim).values.astype(np.float32)
    T, ny, nx = arr.shape
    X = arr.reshape(T, ny * nx)
    t_days = to_epoch_days(da["time"].values)
    w0 = to_epoch_days(ds.attrs.get("window_start", str(pd.Timestamp(da["time"].values[0]).date())))[0]

    any_obs = np.isfinite(X).any(0)
    names = DATE_BANDS + ["season_length", "ndvi_min", "ndvi_max", "ndvi15",
                          "n_obs", "max_gap_days", "last_obs", "qc"]
    flat = {n: np.full(ny * nx, np.nan, np.float32) for n in names}
    cols = np.where(any_obs)[0]
    for s in range(0, cols.size, batch_size):
        cc = cols[s:s + batch_size]
        res = extract_phenology(X[:, cc], t_days, w0, overrides)
        for n in names:
            flat[n][cc] = res[n]

    out = xr.Dataset({n: ((y_dim, x_dim), flat[n].reshape(ny, nx)) for n in names},
                     coords={y_dim: da[y_dim].values, x_dim: da[x_dim].values})
    out.attrs.update(ds.attrs)
    out.attrs["spatial_dims"] = [y_dim, x_dim]
    out.attrs["date_units"] = "days since 1970-01-01"
    qc = flat["qc"][np.isfinite(flat["qc"])]
    n_rice = int((qc < 10).sum())
    print(f"  phenology: {n_rice}/{cols.size} crop pixels with a rice cycle "
          f"({int((qc == 0).sum())} complete, {int((qc == 1).sum())} past peak, "
          f"{int((qc == 2).sum())} pre-peak)")
    return out


# ==================================================================
# Growth-stage map at any date (from the transition-date bands)
# ==================================================================
def classify_stage(ds, as_of):
    """
    Stage class (see STAGE_CLASSES) of every pixel on date `as_of`, derived
    from the transition dates alone — so any number of dates can be mapped
    from one phenology run. -1 = nodata (non-crop / masked), 0 = no rice cycle.
    """
    t = to_epoch_days(as_of)[0]
    qc = ds["qc"].values
    D = {b: ds[b].values for b in DATE_BANDS}
    ge = lambda b: np.isfinite(D[b]) & (t >= D[b])
    stage = np.full(qc.shape, STAGE_NODATA, np.int16)
    has = np.isfinite(qc)
    stage[has] = 0
    rice = has & (qc < 10)
    stage[rice] = 1                                         # pre-planting / transplanting
    for code, band in STAGE_START_BAND:                     # later stages overwrite earlier
        stage[rice & ge(band)] = code
    out = xr.DataArray(stage, dims=ds["qc"].dims, coords=ds["qc"].coords,
                       name="growth_stage",
                       attrs={"as_of": str(pd.Timestamp(as_of).date()),
                              "classes": "; ".join(f"{k}={v}" for k, v in STAGE_CLASSES.items()),
                              "nodata": STAGE_NODATA})
    return out


def classify_stage_period(ds, start, end, method="dominant"):
    """
    ONE growth-stage value per pixel for the period [start, end) — a month,
    a half-month, or any span.

    method : 'dominant' -> the stage the crop spends the most days in during
                           the period (computed exactly from interval overlaps)
             'midpoint' -> the stage on the middle day of the period
    Returns int16 (y, x): 0..5 per STAGE_CLASSES, -1 = nodata.
    """
    p0, p1 = pd.Timestamp(start), pd.Timestamp(end)
    if method in ("midpoint", "midmonth"):
        out = classify_stage(ds, p0 + (p1 - p0) / 2)
    else:
        a, b = to_epoch_days(p0)[0], to_epoch_days(p1)[0]
        qc = ds["qc"].values
        # stage k lasts from its start date to the next stage's start date
        starts = [np.full(qc.shape, -np.inf)] + [
            np.where(np.isfinite(ds[band].values), ds[band].values, np.inf)
            for _, band in STAGE_START_BAND]
        codes = [1] + [c for c, _ in STAGE_START_BAND]
        days = []
        for i in range(len(codes)):
            s = starts[i]
            e = np.minimum.reduce(starts[i + 1:]) if i + 1 < len(starts) else np.full(qc.shape, np.inf)
            days.append(np.clip(np.minimum(e, b) - np.maximum(s, a), 0, None))
        days = np.stack(days)
        # ties -> the later stage (the crop has progressed into it)
        pick = len(codes) - 1 - np.argmax(days[::-1], axis=0)
        stage = np.full(qc.shape, STAGE_NODATA, np.int16)
        has = np.isfinite(qc)
        stage[has] = 0
        rice = has & (qc < 10)
        stage[rice] = np.array(codes, np.int16)[pick][rice]
        out = xr.DataArray(stage, dims=ds["qc"].dims, coords=ds["qc"].coords,
                           name="growth_stage")
    out.attrs.update({"as_of": str(p0.date()), "period_end": str(p1.date()),
                      "method": method,
                      "classes": "; ".join(f"{k}={v}" for k, v in STAGE_CLASSES.items()),
                      "nodata": STAGE_NODATA})
    return out


def classify_stage_month(ds, month, method="dominant"):
    """classify_stage_period for a calendar month 'YYYY-MM' ('midmonth' =
    stage on the 15th)."""
    m0 = pd.Timestamp(f"{month}-01")
    m1 = m0 + pd.offsets.MonthBegin(1)
    if method == "midmonth":
        out = classify_stage(ds, m0 + pd.Timedelta(days=14))
        out.attrs["method"] = method
    else:
        out = classify_stage_period(ds, m0, m1, method)
    out.attrs["as_of"] = m0.strftime("%Y-%m")
    return out


def season_months(ds):
    """All calendar months covered by the season window, as 'YYYY-MM'."""
    p = pd.period_range(ds.attrs["window_start"], ds.attrs["window_end"], freq="M")
    return [str(x) for x in p]


# ==================================================================
# Spatial clean-up (block-wise, NaN-aware, memory-bounded)
# ==================================================================
def _blockwise(arr, radius, fn, block_rows=1024, pad_value=np.nan):
    H, W = arr.shape
    out = np.empty_like(arr)
    for r0 in range(0, H, block_rows):
        r1 = min(H, r0 + block_rows)
        a0, a1 = max(0, r0 - radius), min(H, r1 + radius)
        blk = arr[a0:a1]
        pad_top = radius - (r0 - a0)
        pad_bot = radius - (a1 - r1)
        blk = np.pad(blk, ((pad_top, pad_bot), (radius, radius)),
                     constant_values=pad_value)
        out[r0:r1] = fn(blk, r1 - r0, W)
    return out


def nan_median_filter(arr, radius=1):
    """(2r+1)^2 median ignoring NaN; NaN centres stay NaN."""
    if not radius:
        return arr
    arr = np.asarray(arr, np.float32)
    k = 2 * radius + 1

    def fn(blk, h, w):
        stack = np.stack([blk[i:i + h, j:j + w] for i in range(k) for j in range(k)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
        return np.where(np.isfinite(stack[len(stack) // 2]), med, np.nan)
    return _blockwise(arr, radius, fn)


def majority_filter(grid, radius=1, nodata=STAGE_NODATA):
    """Vectorised mode filter for class grids; nodata ignored, stays nodata."""
    if not radius:
        return grid
    grid = np.asarray(grid)
    k = 2 * radius + 1
    classes = np.unique(grid[grid != nodata])
    if classes.size == 0:
        return grid

    def fn(blk, h, w):
        stack = np.stack([blk[i:i + h, j:j + w] for i in range(k) for j in range(k)])
        counts = np.stack([(stack == c).sum(0) for c in classes])
        mode = classes[np.argmax(counts, axis=0)]
        centre = stack[len(stack) // 2]
        return np.where(centre == nodata, nodata, mode).astype(grid.dtype)
    return _blockwise(grid, radius, fn, pad_value=nodata)


def smooth_dates(ds, radius=1):
    """3x3 NaN-aware median on every date band, then re-derive season length."""
    if not radius:
        return ds
    out = ds.copy()
    for b in DATE_BANDS:
        out[b] = (out[b].dims, nan_median_filter(out[b].values, radius))
    out["season_length"] = out["harvest"] - out["emergence"]
    return out


# ==================================================================
# Export: multi-band int16 COG
# ==================================================================
def to_doy(ds_days, ref_year):
    """epoch days -> day-of-year relative to Jan 1 of ref_year (1 = Jan 1).
    Dates in the previous year are <= 0, dates in the next year are > 365."""
    ref = to_epoch_days(f"{ref_year}-01-01")[0]
    return ds_days - ref + 1


def write_cog(ds, path, ref_year, bands=None, nodata=-32768):
    """
    Write phenology products as one int16 COG.
      date bands  -> DOY relative to Jan 1 of ref_year (see to_doy)
      ndvi_*      -> NDVI x 10000
      others      -> rounded as-is
    Band descriptions + tags carry the encoding so the file is self-describing.
    """
    import os, tempfile
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin

    bands = bands or (DATE_BANDS + ["season_length", "ndvi_min", "ndvi_max",
                                    "ndvi15", "qc", "n_obs", "max_gap_days"])
    y_dim, x_dim = ds.attrs.get("spatial_dims", ["y", "x"])
    if ds[y_dim].values[0] < ds[y_dim].values[-1]:
        ds = ds.sortby(y_dim, ascending=False)
    ys, xs = ds[y_dim].values, ds[x_dim].values
    rx, ry = float(abs(xs[1] - xs[0])), float(abs(ys[1] - ys[0]))
    transform = from_origin(xs[0] - rx / 2, ys[0] + ry / 2, rx, ry)

    data = np.empty((len(bands), ys.size, xs.size), np.int16)
    for i, b in enumerate(bands):
        v = ds[b].values.astype(np.float64)
        if b in DATE_BANDS:
            v = to_doy(v, ref_year)
        elif b.startswith("ndvi"):
            v = v * 10000.0
        v = np.where(np.isfinite(v), np.clip(np.round(v), -32767, 32767), nodata)
        data[i] = v.astype(np.int16)

    profile = dict(driver="GTiff", width=xs.size, height=ys.size, count=len(bands),
                   dtype="int16", crs="EPSG:4326", transform=transform,
                   nodata=nodata, tiled=True, blockxsize=512, blockysize=512,
                   compress="DEFLATE")
    tmp = os.path.join(tempfile.gettempdir(), "_plain_" + os.path.basename(path))
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, b)
        dst.update_tags(
            date_encoding=f"day of year relative to {ref_year}-01-01 (1 = Jan 1; "
                          f"<=0 previous year, >365 next year)",
            ndvi_encoding="NDVI x 10000",
            qc_codes="; ".join(f"{k}={v}" for k, v in QC_DESCRIPTION.items()),
            window=f"{ds.attrs.get('window_start')}..{ds.attrs.get('window_end')}",
            method="S2 NDVI, SG smoothing, 15% amplitude threshold + derivatives")
    rio_copy(tmp, path, driver="COG", compress="DEFLATE", blocksize=512,
             overview_resampling="nearest")
    os.remove(tmp)
    return path


def write_stage_cog(stage_da, path):
    """Single-band int16 COG of a classify_stage() map (nodata -1)."""
    import os, tempfile
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin

    da = stage_da
    if da["y"].values[0] < da["y"].values[-1]:
        da = da.sortby("y", ascending=False)
    ys, xs = da["y"].values, da["x"].values
    rx, ry = float(abs(xs[1] - xs[0])), float(abs(ys[1] - ys[0]))
    profile = dict(driver="GTiff", width=xs.size, height=ys.size, count=1,
                   dtype="int16", crs="EPSG:4326",
                   transform=from_origin(xs[0] - rx / 2, ys[0] + ry / 2, rx, ry),
                   nodata=STAGE_NODATA, tiled=True, blockxsize=512, blockysize=512)
    tmp = os.path.join(tempfile.gettempdir(), "_plain_" + os.path.basename(path))
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(da.values.astype(np.int16), 1)
        dst.set_band_description(1, f"growth_stage {da.attrs.get('as_of')}")
        dst.update_tags(as_of=da.attrs.get("as_of"), classes=da.attrs.get("classes"),
                        method=da.attrs.get("method", "date"))
    rio_copy(tmp, path, driver="COG", compress="DEFLATE", blocksize=512,
             overview_resampling="nearest")
    os.remove(tmp)
    return path


def recent_stage(ds, as_of):
    """
    Near-real-time product for a rolling-window phenology run: Dataset with
      growth_stage   stage on `as_of` (classify_stage)
      data_age_days  as_of - date of the newest clear composite (how fresh
                     the stage is; large values = long cloud gap)
      qc             phenology QC code
    """
    st = classify_stage(ds, as_of)
    t = to_epoch_days(as_of)[0]
    age = np.where(np.isfinite(ds["last_obs"].values), t - ds["last_obs"].values, np.nan)
    out = xr.Dataset({"growth_stage": st,
                      "data_age_days": (st.dims, age.astype(np.float32)),
                      "qc": ds["qc"]})
    out.attrs.update(ds.attrs)
    out.attrs["as_of"] = str(pd.Timestamp(as_of).date())
    return out


def write_recent_cog(rec, path):
    """3-band int16 COG of recent_stage(): growth_stage, data_age_days, qc
    (nodata -32768; growth_stage keeps -1 = non-crop as in the stage maps)."""
    import os, tempfile
    import rasterio
    from rasterio.shutil import copy as rio_copy
    from rasterio.transform import from_origin

    if rec["y"].values[0] < rec["y"].values[-1]:
        rec = rec.sortby("y", ascending=False)
    ys, xs = rec["y"].values, rec["x"].values
    rx, ry = float(abs(xs[1] - xs[0])), float(abs(ys[1] - ys[0]))
    nodata = -32768
    bands = ["growth_stage", "data_age_days", "qc"]
    data = np.empty((3, ys.size, xs.size), np.int16)
    for i, b in enumerate(bands):
        v = rec[b].values.astype(np.float64)
        data[i] = np.where(np.isfinite(v), np.clip(np.round(v), -32767, 32767),
                           nodata).astype(np.int16)
    profile = dict(driver="GTiff", width=xs.size, height=ys.size, count=3,
                   dtype="int16", crs="EPSG:4326",
                   transform=from_origin(xs[0] - rx / 2, ys[0] + ry / 2, rx, ry),
                   nodata=nodata, tiled=True, blockxsize=512, blockysize=512)
    tmp = os.path.join(tempfile.gettempdir(), "_plain_" + os.path.basename(path))
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data)
        for i, b in enumerate(bands, 1):
            dst.set_band_description(i, b)
        dst.update_tags(as_of=rec.attrs.get("as_of"),
                        classes="; ".join(f"{k}={v}" for k, v in STAGE_CLASSES.items()),
                        qc_codes="; ".join(f"{k}={v}" for k, v in QC_DESCRIPTION.items()),
                        window=f"{rec.attrs.get('window_start')}..{rec.attrs.get('window_end')}")
    rio_copy(tmp, path, driver="COG", compress="DEFLATE", blocksize=512,
             overview_resampling="nearest")
    os.remove(tmp)
    return path
