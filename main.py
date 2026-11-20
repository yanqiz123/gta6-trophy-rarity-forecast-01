"""Trophy rarity forecaster.

Reads make.fun case files (``*.in`` JSON) from an input directory and
writes one ``*.out`` line per case.  Each output row gives, for every
trophy, quantile forecasts of its earn rate (in percent) at the case's
target moment.

Method, in brief: work on the logit scale, rescaled by a per-tier
ceiling so that common and rare trophies share one geometry.  A short
least-squares trend in that ceiling-logit space is blended with a
tier/secret prior slope, the forecast centre is shifted when the last
observation is stale (a dilution effect), and the spread combines the
fit residual, a fixed refresh term, the dilution spread, and an
empirical random-walk scale estimated from the case's own successive
snapshot steps (median absolute deviation of the per-sqrt-day logit
steps).  Pure standard library, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Keep sibling imports possible when the runner uses ``python3 -P``.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Highest plausible earn rate (percent) for each tier.
TIER_CEILING = {
    "platinum": 2.0,
    "gold": 25.0,
    "silver": 40.0,
    "bronze": 75.0,
}
FALLBACK_CEILING = 75.0

# Prior trend slopes, keyed by (tier, secret).
PRIOR_SLOPE = {
    ("platinum", True): 0.15,
    ("platinum", False): 0.15,
    ("gold", True): 0.5,
    ("gold", False): 0.4,
    ("silver", True): 0.9,
    ("silver", False): 0.5,
    ("bronze", True): 1.6,
    ("bronze", False): 0.9,
}
FALLBACK_PRIOR = 0.9

STALE_HOURS = 2.0
DILUTION_SHIFT = -1.3
DILUTION_SPREAD = 0.4
REFRESH_SPREAD = 0.30
DEFAULT_RESID = 0.35
MAX_SLOPE = 4.0
SPREAD_MIN = 0.30
SPREAD_MAX = 1.50
FIT_WINDOW_HOURS = 24.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

def clamp(x):
    """Clamp a percent value into [0.01, 99.99]."""
    return min(99.99, max(0.01, x))


def logit(pct):
    """Log-odds of a percent value (clamped first)."""
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    """Inverse logit, returning a percent; stable for large |lg|."""
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimal places, half up; whole values become ints."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def median(vals):
    """Ordinary median, or None for an empty sequence."""
    if not vals:
        return None
    ordered = sorted(vals)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def ols(xs, ys):
    """Least-squares fit y ~ x.

    Returns (slope, intercept, resid_sd), or None when there are too
    few points or the x values are (numerically) constant.  The
    residual standard deviation uses the n-2 denominator.
    """
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx < 1e-9:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        ss = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        resid_sd = math.sqrt(ss / (n - 2))
    else:
        resid_sd = 0.0
    return slope, intercept, resid_sd


# Acklam's rational approximation to the standard normal quantile.
_A = (
    -3.969683028665376e+01,
    2.209460984245205e+02,
    -2.759285984469534e+02,
    1.383577518672690e+02,
    -3.066479806614716e+01,
    2.506628277459239e+00,
)
_B = (
    -5.447609879822406e+01,
    1.615858368580409e+02,
    -1.556989798598866e+02,
    6.680131188771972e+01,
    -1.328068155288572e+01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e+00,
    -2.549732539343734e+00,
    4.374664141464968e+00,
    2.938163982698783e+00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e+00,
    3.754408661907416e+00,
)


def _horner(coeffs, x):
    """Evaluate a polynomial whose coefficients are highest-degree first."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def norm_quantile(p):
    import math as _math
    _a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    _b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01)
    _c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    _d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    def _horner(_coeffs, _x):
        _v = _coeffs[0]
        for _coeff in _coeffs[1:]:
            _v = _v * _x + _coeff
        return _v
    _p = p
    if _p < 0.02425:
        _q = _math.sqrt(-2.0 * _math.log(_p))
        return _horner(_c, _q) / (_horner(_d, _q) * _q + 1.0)
    if _p > 1.0 - 0.02425:
        _q = _math.sqrt(-2.0 * _math.log(1.0 - _p))
        return -(_horner(_c, _q) / (_horner(_d, _q) * _q + 1.0))
    _q = _p - 0.5
    _r = _q * _q
    return _horner(_a, _r) * _q / (_horner(_b, _r) * _r + 1.0)


def parse_time(text):
    """Parse an ISO timestamp, accepting a trailing 'Z'."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def release_moment(case, checkpoint):
    """Release instant: game.release_date, else first read, else checkpoint."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            moment = datetime.fromisoformat(str(raw))
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=timezone.utc)
            return moment
        except ValueError:
            pass
    history = case.get("history") or []
    if history:
        return parse_time(history[0]["at"])
    return checkpoint


def hours_between(start, end):
    return (end - start).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

def prior_slope_for(tier, secret):
    if (tier, secret) in PRIOR_SLOPE:
        return PRIOR_SLOPE[(tier, secret)]
    if (tier, False) in PRIOR_SLOPE:
        return PRIOR_SLOPE[(tier, False)]
    return FALLBACK_PRIOR


def ceiling_logit(rate, ceiling):
    """Map a percent rate into the logit space of its tier ceiling."""
    return logit(100.0 * min(rate, ceiling * 0.999) / ceiling)


def forecast_case(case, name):
    levels = case.get("levels") or list(DEFAULT_LEVELS)
    trophies = case.get("trophies") or []
    history = case.get("history") or []

    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])
    release = release_moment(case, checkpoint)

    def xcoord(moment):
        hrs = hours_between(release, moment)
        return math.log(1.0 + max(-0.99, hrs))

    # -- 1. drop consecutive duplicate reads -------------------------------
    snapshots = []  # list of (moment, players_tracked, rates list)
    for read in history:
        rates = list(read.get("rates") or [])
        key = (read.get("players_tracked"), tuple(rates))
        if snapshots and snapshots[-1][1] == key[0] and tuple(snapshots[-1][2]) == key[1]:
            continue
        snapshots.append((parse_time(read["at"]), read.get("players_tracked"), rates))
    if not snapshots:
        raise ValueError(f"{name} has no history")

    # -- 2/3. time coordinates and horizon ---------------------------------
    x_target = xcoord(target)
    x_last = xcoord(parse_time(history[-1]["at"]))
    dx = x_target - x_last
    horizon_minutes = case["horizon_minutes"]
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    # -- 4. staleness and the dilution it implies ---------------------------
    last_snap_moment = snapshots[-1][0]
    stale = hours_between(last_snap_moment, checkpoint) >= STALE_HOURS
    if stale:
        dil_weight = max(0.0, 1.0 - horizon_hours / 24.0)
    else:
        dil_weight = 0.0
    dil_shift = DILUTION_SHIFT * dil_weight
    dil_spread = DILUTION_SPREAD * dil_weight

    # -- 5. fitting window ---------------------------------------------------
    window_start = checkpoint - timedelta(hours=FIT_WINDOW_HOURS)
    window = [snap for snap in snapshots if snap[0] >= window_start]
    if len(window) < 3:
        window = list(snapshots)

    # -- 6. last observed rate per trophy ------------------------------------
    last_rate = []
    for idx in range(len(trophies)):
        found = None
        for _moment, _players, rates in reversed(snapshots):
            if idx < len(rates) and rates[idx] is not None:
                found = rates[idx]
                break
        last_rate.append(found)

    # Fill gaps from the same tier: upper median of that tier's observed
    # last rates (index len//2 of the sorted list); 0.5 if a tier has none.
    # Peer sets are read from the original observations only, so a filled
    # value never feeds another trophy's fill.
    observed = list(last_rate)
    for idx, trophy in enumerate(trophies):
        if last_rate[idx] is not None:
            continue
        tier = trophy.get("tier")
        peers = sorted(
            observed[j]
            for j, other in enumerate(trophies)
            if other.get("tier") == tier and observed[j] is not None
        )
        last_rate[idx] = peers[len(peers) // 2] if peers else 0.5

    # -- 10 (global part). empirical step scale ------------------------------
    steps = []
    for (ta, _pa, ra), (tb, _pb, rb) in zip(snapshots, snapshots[1:]):
        dt_days = (tb - ta).total_seconds() / 86400.0
        if dt_days <= 0:
            continue
        root_dt = math.sqrt(dt_days)
        for idx in range(len(trophies)):
            if idx < len(ra) and idx < len(rb) and ra[idx] is not None and rb[idx] is not None:
                steps.append((logit(rb[idx]) - logit(ra[idx])) / root_dt)
    if len(steps) >= 10:
        med_step = median(steps)
        mad = median([abs(s - med_step) for s in steps])
        step_scale = 1.4826 * mad
    else:
        step_scale = 0.0
    rw = step_scale * math.sqrt(horizon_days)

    floor_logit = logit(0.01)
    rows = []

    for idx, trophy in enumerate(trophies):
        r0 = last_rate[idx]

        # -- 7. essentially everyone already has it -------------------------
        if r0 >= 99.995:
            rows.append({"id": trophy["id"], "q": [99.5, 99.99, 99.99, 99.99, 99.99]})
            continue

        tier = trophy.get("tier")
        ceiling = TIER_CEILING.get(tier, FALLBACK_CEILING)
        prior = prior_slope_for(tier, bool(trophy.get("secret")))

        # -- 8. trend fit in ceiling-logit space -----------------------------
        fit_xs = []
        fit_ys = []
        for moment, _players, rates in window:
            if idx < len(rates) and rates[idx] is not None:
                fit_xs.append(xcoord(moment))
                fit_ys.append(ceiling_logit(rates[idx], ceiling))

        slope = prior
        resid = DEFAULT_RESID
        fit = ols(fit_xs, fit_ys) if len(fit_xs) >= 3 else None
        if fit is not None and (max(fit_ys) - min(fit_ys)) > 0.01:
            fit_slope, _intercept, fit_resid = fit
            n = len(fit_xs)
            w = n / (n + 3.0)
            slope = w * fit_slope + (1.0 - w) * prior
            resid = min(0.60, max(0.10, fit_resid))
        slope = min(MAX_SLOPE, max(-MAX_SLOPE, slope))

        # -- 9. forecast centre ----------------------------------------------
        base = ceiling_logit(r0, ceiling)
        med_ceil = base + slope * dx + dil_shift
        # ceiling * sigmoid(med_ceil), computed stably, then back to logit.
        if med_ceil >= 0:
            centre_pct = ceiling / (1.0 + math.exp(-med_ceil))
        else:
            e = math.exp(med_ceil)
            centre_pct = ceiling * e / (1.0 + e)
        median_logit = max(logit(centre_pct), floor_logit)

        # -- 10 (per trophy). spread ------------------------------------------
        spread = math.sqrt(
            rw * rw + resid * resid + REFRESH_SPREAD ** 2 + dil_spread * dil_spread
        )
        spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))

        # -- 11. quantiles, forced non-decreasing ------------------------------
        qs = []
        for level in levels:
            z = norm_quantile(level)
            value = round4(clamp(inv_logit(median_logit + z * spread)))
            if qs and value < qs[-1]:
                value = qs[-1]
            qs.append(value)
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    # The output directory may be a mounted volume: create it if needed,
    # never remove it, and only write the per-case files into it.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case, path.name)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
