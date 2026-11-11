#!/usr/bin/env python3
"""Roblox Up-and-Coming forecast entry for make.fun
competition roblox-up-and-coming-forecast-01.

Reads the chart history in $MAKEFUN_INPUT_DIR (default /input):
  - history.ndjson: one JSON snapshot per line, oldest first; each has
    `at` (ISO 8601 UTC) and `up_and_coming`, the chart rows in order
    (`rank`, `universe_id`, `root_place_id`, `name`, `playing`, ...).
  - task.json: `places` (10) and `output` ("forecast.json").
Writes one forecast to $MAKEFUN_OUTPUT_DIR (default /output)/forecast.json:
  {"probabilities": [{"universe_id": int, "place": int, "p": number}, ...],
   "top10": [10 distinct universe ids in order]}
Each place's named probabilities sum to 0.95; the remaining 0.05 is left
unnamed on purpose: the scorer shares a place's unnamed remainder equally
over every game on that read's chart that was not named at that place, so
brand-new games (published after the history ends) still score far above
the 0.0001 floor instead of hitting it.

Model (stdlib only, no network, no randomness, no wall clock):
  1. Empirical rank-transition kernels from the history itself: for every
     pair of snapshots 0-72h apart, binned by horizon (0-12h, 12-36h,
     36-72h), count where a game at rank r ended up (place 1-10 or out of
     the top 10), pair-weighted by recency. The bins are mixed in
     proportion to the hours of the 72h scored window they cover
     (12/72, 24/72, 36/72). Each kernel is smoothed with a persistence
     prior (a game is most likely to stay near its rank) so sparse
     ranks and short histories still give a sane distribution.
  2. Recency-weighted place occupancy (marginal): how often each game
     actually held each place, half-life 96h. Blended in at 30% so games
     that repeatedly held a place keep credit even when they have just
     slipped off it, and games absent from the last snapshot are covered.
  3. No trend boost: a rank-velocity multiplier was in the first version
     and was removed after walk-forward backtests (synthetic chart
     histories across stable/typical/volatile/trending regimes) showed
     it consistently cost log score — recent rank velocity is mostly
     noise at these horizons and the kernels already price movement.
  4. Per place: keep the top 36 candidates, floor each at 0.002,
     cap each at 0.70, normalise to sum 0.95. No probability is 0 or
     near 1, which is what the log score rewards: a confident wrong
     call costs ln(0.0001) = -9.21 a place, an honest spread does not.
  5. top10 is picked greedily place by place from the probabilities
     (highest p at place 1, then place 2 among the games not used, ...),
     falling back to expected-place score and then the last chart order.

Determinism: every weight derives from snapshot timestamps in the
input, iteration order is fully sorted, and probabilities are rounded
to 6 decimals with the residual put on the largest row, so two runs on
the same history write byte-identical files.
"""

import json
import math
import os
from datetime import datetime, timezone

TARGET_SUM = 0.95      # named share per place; 0.05 stays unnamed
P_CAP = 0.70           # no game gets more than this at one place
P_FLOOR = 0.002        # every named game gets at least this
MAX_ROWS_PER_PLACE = 36
MAX_TRACKED_RANK = 60  # transitions are tracked from ranks up to this
PRIOR_WEIGHT = 6.0     # persistence-prior pseudo-count per kernel
KERNEL_WEIGHT = 0.70   # transition kernel share of the raw score
MARGINAL_WEIGHT = 0.30  # recency-weighted occupancy share
HORIZON_BINS = ((0.0, 12.0, 12.0), (12.0, 36.0, 24.0), (36.0, 72.0, 36.0))


def parse_time(text):
    """Parse an ISO 8601 timestamp to epoch seconds (float)."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_history(input_dir):
    """Return (task, snapshots) where each snapshot is (epoch, [uid, ...])."""
    task = {"places": 10, "output": "forecast.json"}
    task_path = os.path.join(input_dir, "task.json")
    if os.path.exists(task_path):
        with open(task_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            task.update(loaded)
    snapshots = []
    history_path = os.path.join(input_dir, "history.ndjson")
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    snap = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(snap, dict):
                    continue
                rows = snap.get("up_and_coming") or []
                chart = []
                seen = set()
                rows_sorted = sorted(
                    (r for r in rows if isinstance(r, dict)),
                    key=lambda r: r.get("rank") if isinstance(r.get("rank"), int) else 0,
                )
                for row in rows_sorted:
                    uid = row.get("universe_id")
                    if isinstance(uid, int) and uid not in seen:
                        seen.add(uid)
                        chart.append(uid)
                at = parse_time(snap.get("at"))
                if chart and at is not None:
                    snapshots.append((at, chart))
    snapshots.sort(key=lambda item: item[0])
    return task, snapshots


def persistence_prior(rank):
    """Prior distribution over destinations 0..10 for a game at `rank`.

    Destination 0 means 'out of the top 10'. Unnormalised weights.
    """
    weights = [0.0] * 11
    for place in range(1, 11):
        weights[place] = math.exp(-abs(rank - place) / 3.0)
    if rank <= 10:
        weights[0] = 1.1
    elif rank <= 20:
        weights[0] = 2.2
    else:
        weights[0] = 3.5
    return weights


def build_raw_scores(snapshots, places):
    """Return raw[place][uid] = unnormalised probability mass."""
    n = len(snapshots)
    last_time = snapshots[-1][0]
    rank_of = []  # per snapshot: uid -> rank (1-based)
    for _, chart in snapshots:
        rank_of.append({uid: idx + 1 for idx, uid in enumerate(chart)})

    # --- recency-weighted marginal occupancy per place ---
    marg = [dict() for _ in range(places + 1)]
    marg_total = [0.0] * (places + 1)
    for idx, (at, chart) in enumerate(snapshots):
        age_h = (last_time - at) / 3600.0
        w = 0.5 ** (age_h / 96.0)
        for place in range(1, places + 1):
            if place <= len(chart):
                uid = chart[place - 1]
                marg[place][uid] = marg[place].get(uid, 0.0) + w
                marg_total[place] += w
    marg_norm = [dict() for _ in range(places + 1)]
    for place in range(1, places + 1):
        if marg_total[place] > 0:
            for uid, value in marg[place].items():
                marg_norm[place][uid] = value / marg_total[place]

    # --- empirical transition kernels, per horizon bin ---
    nbins = len(HORIZON_BINS)
    counts = [[[0.0] * 11 for _ in range(MAX_TRACKED_RANK + 1)] for _ in range(nbins)]
    chart_sets = [set(chart) for _, chart in snapshots]
    for i in range(n):
        t_i, chart_i = snapshots[i]
        age_h = (last_time - t_i) / 3600.0
        pair_w = 0.5 ** (age_h / 240.0)
        tracked = [(uid, r) for uid, r in rank_of[i].items() if r <= MAX_TRACKED_RANK]
        for j in range(i + 1, n):
            dt_h = (snapshots[j][0] - t_i) / 3600.0
            if dt_h <= 0:
                continue
            if dt_h > 72.0:
                break
            bin_idx = None
            for b, (lo, hi, _) in enumerate(HORIZON_BINS):
                if lo < dt_h <= hi or (b == 0 and dt_h <= hi):
                    bin_idx = b
                    break
            if bin_idx is None:
                continue
            ranks_j = rank_of[j]
            for uid, r in tracked:
                dest = ranks_j.get(uid, 0)
                if dest > 10:
                    dest = 0
                counts[bin_idx][r][dest] += pair_w
            _ = chart_sets  # sets reserved; rank maps carry the lookup

    prior = [None] + [persistence_prior(r) for r in range(1, MAX_TRACKED_RANK + 1)]
    horizon_total = sum(span for _, _, span in HORIZON_BINS)
    kernel_mix = [[0.0] * 11 for _ in range(MAX_TRACKED_RANK + 1)]
    for r in range(1, MAX_TRACKED_RANK + 1):
        for b, (_, _, span) in enumerate(HORIZON_BINS):
            share = span / horizon_total
            row = counts[b][r]
            denom = sum(row) + PRIOR_WEIGHT * sum(prior[r])
            if denom <= 0:
                continue
            for dest in range(11):
                smoothed = (row[dest] + PRIOR_WEIGHT * prior[r][dest]) / denom
                kernel_mix[r][dest] += share * smoothed

    # --- combine kernel + marginal for every candidate game ---
    last_ranks = rank_of[-1]
    candidates = set(last_ranks)
    for place in range(1, places + 1):
        candidates.update(marg_norm[place])
    raw = [dict() for _ in range(places + 1)]
    for uid in candidates:
        r_last = last_ranks.get(uid)
        for place in range(1, places + 1):
            if r_last is not None and r_last <= MAX_TRACKED_RANK:
                base = KERNEL_WEIGHT * kernel_mix[r_last][place]
            elif r_last is not None:
                base = KERNEL_WEIGHT * math.exp(-abs(r_last - place) / 3.0) * 0.05
            else:
                base = 0.0
            base += MARGINAL_WEIGHT * marg_norm[place].get(uid, 0.0)
            if base > 0:
                raw[place][uid] = base
    return raw


def cap_and_scale(values, target, cap):
    """Scale values to sum to target, no entry above cap (water-filling)."""
    values = list(values)
    fixed = [False] * len(values)
    remaining = target
    for _ in range(len(values) + 1):
        active = [i for i in range(len(values)) if not fixed[i]]
        if not active:
            break
        total = sum(values[i] for i in active)
        if total <= 0:
            share = remaining / len(active)
            for i in active:
                values[i] = share
            break
        scale = remaining / total
        over = [i for i in active if values[i] * scale > cap]
        if not over:
            for i in active:
                values[i] *= scale
            break
        for i in active:
            values[i] *= scale
        for i in over:
            remaining -= cap
            values[i] = cap
            fixed[i] = True
    return values


def build_forecast(task, snapshots):
    places = int(task.get("places") or 10)
    if not snapshots:
        return {"probabilities": [], "top10": []}
    raw = build_raw_scores(snapshots, places)
    rows = []
    per_place = {}
    for place in range(1, places + 1):
        ranked = sorted(raw[place].items(), key=lambda kv: (-kv[1], kv[0]))
        ranked = ranked[:MAX_ROWS_PER_PLACE]
        if not ranked:
            per_place[place] = []
            continue
        values = cap_and_scale(
            [max(value, P_FLOOR) for _, value in ranked], TARGET_SUM, P_CAP
        )
        rounded = [round(v, 6) for v in values]
        residual = round(TARGET_SUM - sum(rounded), 6)
        if rounded and residual != 0.0:
            biggest = max(range(len(rounded)), key=lambda i: rounded[i])
            adjusted = round(rounded[biggest] + residual, 6)
            if 0.0 < adjusted <= P_CAP + 0.001:
                rounded[biggest] = adjusted
        place_rows = []
        for (uid, _), p in zip(ranked, rounded):
            if p > 0:
                place_rows.append({"universe_id": uid, "place": place, "p": p})
                rows.append(place_rows[-1])
        per_place[place] = place_rows

    # --- top10: greedy by place, then expected score, then last chart ---
    used = []
    used_set = set()
    by_place = {
        place: sorted(per_place.get(place, []), key=lambda r: (-r["p"], r["universe_id"]))
        for place in range(1, places + 1)
    }
    for place in range(1, places + 1):
        for row in by_place[place]:
            if row["universe_id"] not in used_set:
                used.append(row["universe_id"])
                used_set.add(row["universe_id"])
                break
    if len(used) < places:
        expected = {}
        for place in range(1, places + 1):
            for row in per_place.get(place, []):
                expected[row["universe_id"]] = expected.get(row["universe_id"], 0.0) + row["p"] / place
        for uid, _ in sorted(expected.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(used) >= places:
                break
            if uid not in used_set:
                used.append(uid)
                used_set.add(uid)
    if len(used) < places:
        for uid in snapshots[-1][1]:
            if len(used) >= places:
                break
            if uid not in used_set:
                used.append(uid)
                used_set.add(uid)
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    return {"probabilities": rows, "top10": used[:places]}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    task, snapshots = load_history(input_dir)
    forecast = build_forecast(task, snapshots)
    os.makedirs(output_dir, exist_ok=True)
    out_name = task.get("output") or "forecast.json"
    out_path = os.path.join(output_dir, out_name)
    text = json.dumps(forecast, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
