# GTA VI trophy rarity forecast — make.fun `gta6-trophy-rarity-forecast-01`

Forecasts each Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at
2026-11-26T00:00:00Z, as five quantiles (0.05 / 0.25 / 0.5 / 0.75 / 0.95)
per trophy. One file, Python 3.13, standard library only, no network.

Run it the way the grader does:

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `<id>.in` in the input directory and writes `<id>.out` to the
output directory. Runtime is ~0.1 s per case on one CPU (limit: 8 s).

## The model

Exophase refreshes a game's counts on its own schedule, hours apart, so the
recorded history is a step function: long flat stretches, then a jump. Each
refresh also *dilutes* every rate, because the newly tracked players have
earned far less than the players tracked before them. Between refreshes the
rates only grow, as players work through the game. The forecast, per trophy,
on the logit scale:

1. **Collapse the history to distinct snapshots.** A read identical to the
   previous one is the same Exophase page, not new information.
2. **Fit a trend.** Least squares of `logit(rate / tier-ceiling)` against
   `ln(1 + hours since release)`, on the distinct snapshots of the last 24
   hours (all of them if there are fewer than three). Fitting in
   ceiling-logit space makes growth damp itself as a rate nears its tier
   ceiling instead of compounding forever.
3. **Shrink the slope** toward a prior for the trophy's kind — secret/story
   trophies grow fast, public activity and grind trophies slowly, platinum
   slowest — with the weight on the fit growing with the number of distinct
   snapshots. A trophy with no usable fit gets the prior.
4. **Anchor at the last observed rate** and move it along the shrunk slope
   to the target. The median can approach its tier ceiling (bronze 75%,
   silver 40%, gold 25%, platinum 2%) but never pass it.
5. **Spread honestly.** A random-walk term growing with √horizon, the fit's
   residual scatter, and a refresh term for the jumps the target window will
   contain — about 1.0 logit units of spread for the 4.5-day final horizon.
6. **Short-horizon refresh guard.** When the last snapshot has been unchanged
   for 2+ hours and the horizon is short, a refresh is overdue and the next
   one dilutes the rates (launch week's first refresh multiplied tracked
   players sixfold, 638 → 3,868, and cut most rates to about a fifth). The
   median shifts down ~1 logit unit and the spread widens; the effect fades
   to nothing over a 24-hour horizon, where progression dominates.
7. **A trophy at 100% stays ~100%.** Every tracked player has it, and the
   launch-night refresh proved newcomers have it too.

## Why this beats the house baseline

The baseline says every rate stays where the last read put it. Over a week
that misses badly: easy story trophies climb from ~1% into the tens of
percent while grind trophies barely move. The trend fit, the shrinkage and
the ceilings are what carry the forecast.

## Local checks (not part of the entry)

The kit ships no scorer and no practice answers, so the practice loss below
is against answers reconstructed from the aggregates the rules state (the
04:30Z refresh: 3,868 players, 4,276 trophies earned; every rate but the
100% one lower). That reconstruction reproduces the rules' stated baseline
figures (loss ~1.5, typical miss ~0.68, coverage 0), so it is a fair proxy,
not the server's answers. A synthetic final-like case (plausible Nov 19–21
history, Nov 26 truth from a saturating trajectory of a different family
than the fit) checks the machinery on final-shaped data.

| check | this entry | house baseline (starter) |
|---|---|---|
| practice p00, pinball loss (proxy answers) | 0.68 | 1.40 |
| practice p01, pinball loss (proxy answers) | 0.63 | 1.40 |
| synthetic final, pinball loss | 0.22 | 0.31 |
| runtime per case | ~0.07 s | ~0.07 s |

The practice cases are a smoke check only: their history is one repeated
read, so no trend can be fitted there and nothing was tuned on them.

## Files

- `main.py` — the whole entry.
- `LICENSE` — MIT.
