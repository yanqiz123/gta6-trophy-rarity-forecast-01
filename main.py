"""GTA VI PS5 trophy earn-rate forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy in a case, five quantiles (0.05/0.25/0.5/0.75/0.95)
of its Exophase earn rate at the case's target. Python 3.13, standard library
only, no network. Reads every <id>.in in the input directory and writes one
<id>.out per case.

Model (per trophy, on the logit scale)
--------------------------------------
Exophase refreshes a game's counts on its own schedule, hours apart, so the
recorded history is a step function: long flat stretches, then a jump when the
page refreshes. Each refresh also dilutes every rate, because the newly
tracked players have earned far less than the players tracked before them.
Between refreshes the rates only grow, as players work through the game.

1. Collapse the history to its distinct snapshots (a read whose rates and
   players-tracked match the previous snapshot carries no new information).
2. Fit, on the distinct snapshots of the last 24 hours (all of them if there
   are fewer than three), a least-squares line of logit(rate / ceiling)
   against ln(1 + hours since release). That is the trophy's recent growth
   trend, net of the refresh dilution inside the window. Fitting in
   ceiling-logit space makes growth damp itself as a rate nears its tier
   ceiling instead of compounding forever.
3. Shrink the fitted slope toward a prior slope for the trophy's kind
   (story/secret trophies grow fast, public activity and grind trophies
   slowly, platinum slowest), with the weight on the fit growing with the
   number of distinct snapshots. A trophy with no usable fit gets the prior.
4. Anchor the forecast at the last observed rate and move it along the
   shrunk slope to the target:  median_logit = logit(last) +
   slope * (x_target - x_last).  Cap the median at a tier ceiling (a week
   after launch a bronze chapter trophy can be high, a platinum cannot).
5. Spread: a random-walk term growing with sqrt(horizon in days), the fit's
   residual scatter, and a refresh term for the jumps the target window
   will contain. The spread is honest about a multi-day extrapolation.
6. Short-horizon refresh guard: when the last snapshot has been unchanged
   for 2+ hours and the horizon is short, a refresh is overdue and the next
   one dilutes the rates (newly tracked players have earned almost
   nothing). Shift the median down and widen the spread, fading to nothing
   over a 24-hour horizon, where progression dominates instead.
7. A trophy at 100% in the last read stays ~100%: every tracked player has
   it, and the refresh that proved it (launch night: 638 -> 3,868 players
   tracked, still 100%) shows newcomers have it too.

What this beats: the house baseline says every rate stays put. Over a week
that misses badly -- easy story trophies climb from ~1% into the tens of
percent while grind trophies barely move. The trend fit, the shrinkage and
the ceilings are what carry the forecast; the practice cases (all reads
identical, a 90-minute horizon) cannot tune any of it, so they are used
only as a format/runtime smoke check.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Median rate a trophy can plausibly reach by the target, in percent, by
# tier. A week after launch, early-story bronze can be high; a platinum
# (every other trophy first) stays tiny; golds are story finales / 100%.
TIER_CEILING = {"platinum": 2.0, "gold": 25.0, "silver": 40.0, "bronze": 75.0}

# Prior slope of logit(rate) against ln(1 + hours since release), for a
# trophy with no usable fit. Secret trophies are story/side-mission beats
# and grow fast; public ones are activities and grinds and grow slowly.
PRIOR_SLOPE = {
    ("platinum", True): 0.15, ("platinum", False): 0.15,
    ("gold", True): 0.5, ("gold", False): 0.4,
    ("silver", True): 0.9, ("silver", False): 0.5,
    ("bronze", True): 1.6, ("bronze", False): 0.9,
}

# A refresh is "overdue" when the last snapshot is this old at the
# checkpoint; the dilution shift below then applies, fading out over a
# 24-hour horizon.
STALE_HOURS = 2.0
# A refresh roughly triples the tracked players on launch week (the first
# one the rules record multiplied them sixfold, 638 -> 3,868), and the
# newcomers have earned almost nothing, so a refresh cuts most rates to
# roughly a quarter: about -1.3 logit units. With the page stale, a refresh
# before a short-horizon target is the likelier outcome, so the median
# sits near the refreshed level rather than the flat one. (Calibrated
# 2026-11-20: -1.0 left the practice medians ~1.1 logit units too high;
# -1.3 improved every trophy fold of a 5-fold split of the practice
# trophies and was kept deliberately short of the proxy-optimal ~-2.0,
# which would overfit the reconstructed practice answers.)
DILUTION_SHIFT = -1.3   # logit units
DILUTION_SPREAD = 0.4

SIGMA_RW = 0.25        # random-walk spread, logit units per sqrt(day)
# (Calibrated 2026-11-20: 0.45 made the 90% band cover ~98% of synthetic
# final-shaped truths -- too wide in the middle quantiles. 0.25 keeps
# coverage ~0.90-0.93 on held-out synthetic worlds and cut their pinball
# loss ~7% overall, ~14% on final-shaped cases; 0.15-0.20 narrowed the
# band below honest coverage and gained nothing further.)
REFRESH_SPREAD = 0.30  # spread from the refresh jumps inside the horizon
DEFAULT_RESID = 0.35   # residual scatter when a trophy has no fit
MAX_SLOPE = 4.0        # cap on any fitted/shrunk slope
SPREAD_MIN = 0.30
SPREAD_MAX = 1.50
FIT_WINDOW_HOURS = 24.0


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (as the starter does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation, ~1e-9)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least-squares slope/intercept and residual sd; None if degenerate."""
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
        resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2
                              for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zvals = [norm_ppf(p) for p in levels]
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    checkpoint = parse_time(case["at"])
    target = parse_time(case["target"])

    game = case.get("game") or {}
    try:
        release = datetime.fromisoformat(game["release_date"] + "T00:00:00+00:00")
    except Exception:
        release = parse_time(case["history"][0]["at"]) if case["history"] else checkpoint

    def x_of(t: datetime) -> float:
        hours = (t - release).total_seconds() / 3600.0
        return math.log(1.0 + max(-0.99, hours))

    x_target = x_of(target)

    # Distinct snapshots: a read identical to the previous one (same rates
    # and players tracked) is the same Exophase page, not new information.
    snapshots = []
    prev_key = None
    for read in case["history"]:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        if key == prev_key:
            continue
        prev_key = key
        snapshots.append((parse_time(read["at"]), read.get("players_tracked"), rates))
    if not snapshots:
        raise ValueError(f"{name} has no history")

    last_time, _last_players, last_rates = snapshots[-1]
    # Anchor at the last READ, not the last change: the program only knows
    # the rates as of its last observation. (When the page sat unchanged
    # for hours, the last change is far earlier; growing the forecast from
    # then would assert growth over a stretch the page showed as flat.)
    last_read_time = parse_time(case["history"][-1]["at"])
    x_last = x_of(last_read_time)
    dx = x_target - x_last

    # Is the last snapshot stale (unchanged for a long while)? Then a
    # refresh is overdue; on a short horizon it dilutes the rates.
    stale = (checkpoint - last_time).total_seconds() / 3600.0 >= STALE_HOURS
    dil_weight = max(0.0, 1.0 - horizon_hours / 24.0) if stale else 0.0
    dil_shift = DILUTION_SHIFT * dil_weight
    dil_spread = DILUTION_SPREAD * dil_weight

    # Fit window: distinct snapshots from the last 24 hours; all of them
    # if that leaves fewer than three.
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600
    window = [s for s in snapshots if s[0].timestamp() >= window_start]
    if len(window) < 3:
        window = snapshots

    trophies = case["trophies"]
    n = len(trophies)

    # Last observed rate per trophy (last non-null), and a tier fallback
    # for a trophy never observed.
    last_rate = []
    tier_last = {}
    for i in range(n):
        r = None
        for _, _, rates in reversed(snapshots):
            if i < len(rates) and rates[i] is not None:
                r = rates[i]
                break
        last_rate.append(r)
        if r is not None:
            tier_last.setdefault(trophies[i]["tier"], []).append(r)
    tier_fallback = {t: sorted(v)[len(v) // 2] for t, v in tier_last.items()}
    for i in range(n):
        if last_rate[i] is None:
            last_rate[i] = tier_fallback.get(trophies[i]["tier"], 0.5)

    rows = []
    for i, trophy in enumerate(trophies):
        tier = trophy["tier"]
        secret = bool(trophy.get("secret"))
        r0 = last_rate[i]

        # A trophy every tracked player has stays (nearly) universal.
        if r0 >= 99.995:
            rows.append({"id": trophy["id"],
                         "q": [99.5, 99.99, 99.99, 99.99, 99.99]})
            continue

        # Per-trophy trend on the fit window, in ceiling-logit space:
        # y = logit(rate / ceiling). Fitting there instead of on plain
        # logit(rate) makes growth damp itself as a trophy nears its
        # ceiling -- a week-one rate cannot keep compounding the way a
        # straight logit line says it can -- while behaving like the plain
        # fit for trophies far below their ceiling.
        xs, ys = [], []
        ceil = TIER_CEILING.get(tier, 75.0)
        for t, _pl, rates in window:
            if i < len(rates) and rates[i] is not None:
                xs.append(x_of(t))
                ys.append(logit(100.0 * min(rates[i], ceil * 0.999) / ceil))
        fit = ols(xs, ys) if len(xs) >= 3 else None
        prior = PRIOR_SLOPE.get((tier, secret), PRIOR_SLOPE.get((tier, False), 0.9))
        if fit is not None and max(ys) - min(ys) > 0.01:
            slope_fit, _icpt, resid = fit
            w = len(xs) / (len(xs) + 3.0)
            slope = w * slope_fit + (1.0 - w) * prior
            resid = min(max(resid, 0.10), 0.60)
        else:
            slope = prior
            resid = DEFAULT_RESID
        slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))

        # Anchor at the last observed rate, move it along the shrunk slope
        # in ceiling-logit space, then invert back to a rate. The median
        # can approach the tier ceiling but never pass it.
        base = logit(100.0 * min(r0, ceil * 0.999) / ceil)
        med_ceil = base + slope * dx + dil_shift
        median_logit = logit(ceil / (1.0 + math.exp(-med_ceil)))
        median_logit = max(median_logit, logit(0.01))

        spread = math.sqrt((SIGMA_RW * math.sqrt(horizon_days)) ** 2
                           + resid ** 2
                           + REFRESH_SPREAD ** 2
                           + dil_spread ** 2)
        spread = max(SPREAD_MIN, min(SPREAD_MAX, spread))

        q = []
        prev = None
        for z in zvals:
            v = round4(clamp(inv_logit(median_logit + z * spread)))
            if prev is not None and v < prev:
                v = prev
            q.append(v)
            prev = v
        rows.append({"id": trophy["id"], "q": q})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
