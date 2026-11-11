# Roblox Up-and-Coming Forecast — make.fun roblox-up-and-coming-forecast-01

A program that reads the history of Roblox's Up-and-Coming chart and
forecasts its top 10 for every hour of the 72 hours after entries close,
with honest probabilities per place — built for the make.fun bounty
`roblox-up-and-coming-forecast-01` ($500 pool, log-score ranked).

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

Python 3.13, standard library only, no network, no model calls.
Without the environment variables it reads `/input` and writes
`/output/forecast.json`, as the grader runs it.

## Input

- `history.ndjson` — one JSON snapshot per line, oldest first. Each line
  has `at` (ISO 8601 UTC) and `up_and_coming`: the chart rows in order
  with `rank`, `universe_id`, `root_place_id`, `name`, `playing`,
  `up_votes`, `down_votes`, `genre` (plus `top_trending`,
  `top_playing_now` and `details` context this entry does not need).
- `task.json` — `places` (10), `output` (`forecast.json`), the scored
  `window` and `history_until`.

## Output — `/output/forecast.json`

Exactly two keys:

- `probabilities`: rows `{"universe_id": int, "place": 1..10, "p": number}`,
  at most one row per game per place, at most 36 rows per place here.
  Each place's named probabilities sum to 0.95; the other 0.05 is left
  unnamed on purpose, because the scorer shares a place's unnamed
  remainder equally over every game on that read's chart not named at
  that place — brand-new games still score well above the 0.0001 floor.
- `top10`: exactly 10 distinct universe ids, the guessed top 10 in order.

## How it forecasts

The chart churns hard (the practice history's #1 at the close held #1 in
none of the next 10 reads), so this is not a persistence guess:

1. **Empirical transition kernels.** From the history itself, every pair
   of snapshots 0–72h apart, binned by horizon (0–12h, 12–36h, 36–72h),
   counts where a game at rank *r* ended up (place 1–10 or out), weighted
   by recency and mixed in proportion to the window hours each bin
   covers. Smoothed with a persistence prior for sparse ranks.
2. **Recency-weighted occupancy.** How often each game actually held
   each place (96h half-life), blended in at 30% — it covers games that
   just slipped off a place and games absent from the last snapshot.
3. **No trend boost.** A rank-velocity multiplier shipped in the first
   version and was removed after walk-forward backtests on synthetic
   chart histories (stable/typical/volatile/trending regimes) showed it
   consistently cost log score; the kernels already price movement.
4. **Calibration for the log score.** Per place, top-36 candidates are
   floored at 0.002, capped at 0.70, and normalised to 0.95. Nothing is
   0, nothing is near 1 — a confident wrong call costs −9.21 a place,
   an honest spread does not.
5. **top10** is picked greedily place by place from those probabilities,
   with expected-place score and the last chart as fallbacks.

No randomness and no wall clock: every weight derives from the input's
own timestamps, all ordering is sorted, and probabilities are rounded
to 6 decimals, so two runs on the same history are byte-identical.

## License

MIT — see `LICENSE`. Copyright (c) 2026 George Agbe.
