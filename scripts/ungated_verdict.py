"""What the verification step is worth, measured on the draw instead of on two stocks.

Chapter 6 claims the equal-weight fallback matters because the same learner, left
unverified, loses. That claim rested on the two development stocks. This scores the ungated
arms against their own committees on all twenty names of the draw, and asks three
questions:

  how many configurations lose without verification
  how many of those verification removes outright, by falling back to equal weights
  what verification leaves behind

Both arms are repriced on the next-open convention Chapter 6 uses, so the numbers are
comparable with tab:terminal rather than with the raw report fields.

    TRADING_AGENT_DATA_ROOT=/path/to/data PYTHONPATH=src python scripts/ungated_verdict.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reprice_next_open import adjusted_prices, reprice  # noqa: E402

COST = 10.0 / 1e4
ARMS = ("global_memory", "market_state_memory", "market_state_memory_online")
LABEL = {"global_memory": "global", "market_state_memory": "state",
         "market_state_memory_online": "updated"}
UNGATED = ROOT / "reports" / "ungated20"
REPRICE = ROOT / "reports" / "reprice"
# Sixteen names come from ungated_twenty.py, which re-runs the launcher against the
# frozen view corpus. MTCH, O, RL and TEX hold only the tail of theirs, so they come
# from ungated_replay.py instead: their trained memory is stored and their test window
# is complete, and the replayer reproduces their SEALED arms day for day before it is
# trusted to produce their unverified ones.
NAMES = sorted(p.name.split("__")[0] for p in UNGATED.glob("*__ungated.json"))


def net(rows):
    out, prev = [], 0.0
    for _, gross, exposure in rows:
        out.append(gross - COST * abs(exposure - prev))
        prev = exposure
    return out


def sharpe(v):
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
    return m / sd * math.sqrt(252) if sd else 0.0


opens, closes = adjusted_prices(list(NAMES))


def repriced(rows, t, cal):
    """Next-open repricing needs the day after the last scored one.

    Passing the scored calendar alone silently zeroes every series' final return,
    which on this window is a 3% down day: the first run of this script put the
    unverified equal-weight arm 0.25 Sharpe away from the identical verified one.
    """
    after = [d for d in sorted(opens[t]) if d > cal[-1]][:2]
    return reprice(rows, opens[t], closes[t], cal + after)[:len(cal)]


rows = []
for t in NAMES:
    ung = json.loads((UNGATED / f"{t}__ungated.json").read_text(encoding="utf-8"))
    ver = json.loads((REPRICE / f"{t}__reprice.json").read_text(encoding="utf-8"))
    cal = [r[0] for r in ver["daily_arm_series"]["equal_weight"]]
    up = {a: repriced(r, t, cal) for a, r in ung["daily_arm_series"].items()}
    base_u = sharpe(net(up["equal_weight"]))
    base_v = sharpe(net(ver["daily_arm_series"]["equal_weight"]))
    for a in ARMS:
        du = sharpe(net(up[a])) - base_u
        dv = sharpe(net(ver["daily_arm_series"][a])) - base_v
        identical = [r[2] for r in ver["daily_arm_series"][a]] == \
                    [r[2] for r in ver["daily_arm_series"]["equal_weight"]]
        rows.append((t, LABEL[a], du, dv, identical))

print(f"names {len(NAMES)}: {' '.join(NAMES)}\n")
print(f"{'name':<6}{'arm':<9}{'ungated':>9}{'verified':>10}   what verification did")
print("-" * 62)
for t, a, du, dv, ident in rows:
    if ident:
        did = "removed it (arm equals the committee)"
    elif abs(du - dv) < 5e-4:
        did = "let it through unchanged"
    else:
        did = f"let it act, moved {du:+.3f} to {dv:+.3f}"
    print(f"{t:<6}{a:<9}{du:>+9.3f}{dv:>+10.3f}   {did}")

lose_u = [r for r in rows if r[2] < 0]
lose_v = [r for r in rows if r[3] < 0]
removed = [r for r in lose_u if r[4]]
survived = [r for r in lose_u if not r[4]]
print("\n" + "=" * 62)
print(f"configurations                     {len(rows)}  ({len(NAMES)} names x 3 arms)")
print(f"lose without verification              {len(lose_u)}")
print(f"  of those, verification removes       {len(removed)}")
print(f"  of those, verification lets through  {len(survived)}")
print(f"lose with verification                 {len(lose_v)}")
if lose_u:
    print(f"worst loss without verification        {min(r[2] for r in lose_u):+.3f}")
if lose_v:
    print(f"worst loss with verification           {min(r[3] for r in lose_v):+.3f}")
mu = sum(r[2] for r in rows) / len(rows)
mv = sum(r[3] for r in rows) / len(rows)
print(f"mean delta without verification        {mu:+.4f}")
print(f"mean delta with verification           {mv:+.4f}")

# Verification discards on evidence, not on outcome, so it removes gains as well as losses.
# Both sides are reported: a summary that counted only the removed losses would read as
# though the test could tell them apart in advance.
gain_u = [r for r in rows if r[2] > 0]
gain_removed = [r for r in gain_u if r[4]]
kept = [r for r in rows if not r[4]]
print(f"gain without verification              {len(gain_u)}")
print(f"  of those, verification removes       {len(gain_removed)}")
print(f"mean gain verification removes         "
      f"{sum(r[2] for r in gain_removed)/len(gain_removed):+.4f}"
      if gain_removed else "")
print(f"mean loss verification removes         "
      f"{sum(r[2] for r in removed)/len(removed):+.4f}" if removed else "")
print(f"arms verification keeps                {len(kept)}, mean "
      f"{sum(r[3] for r in kept)/len(kept):+.4f}" if kept else "")
print(f"names where every arm falls back   "
      f"{sum(1 for t in NAMES if all(r[4] for r in rows if r[0] == t))}/{len(NAMES)}")

out = ROOT / "reports" / "ungated20" / "verdict20.json"
out.write_text(json.dumps({
    "names": NAMES,
    "cost_bps_per_side": COST * 1e4,
    "metric": "annualised Sharpe net of cost, minus the same name's equal-weight committee",
    "rows": [{"name": t, "arm": a, "ungated": du, "verified": dv, "gate_removed_it": ident}
             for t, a, du, dv, ident in rows],
    "summary": {
        "configurations": len(rows),
        "lose_without_gate": len(lose_u), "lose_removed": len(removed),
        "lose_survived": len(survived), "lose_with_gate": len(lose_v),
        "gain_without_gate": len(gain_u), "gain_removed": len(gain_removed),
        "worst_without_gate": min((r[2] for r in lose_u), default=0.0),
        "worst_with_gate": min((r[3] for r in lose_v), default=0.0),
        "mean_without_gate": mu, "mean_with_gate": mv,
        "mean_removed_gain": (sum(r[2] for r in gain_removed) / len(gain_removed)
                              if gain_removed else 0.0),
        "mean_removed_loss": (sum(r[2] for r in removed) / len(removed)
                              if removed else 0.0),
        "kept": len(kept),
        "mean_kept": sum(r[3] for r in kept) / len(kept) if kept else 0.0,
        "names_fully_fallen_back": sum(
            1 for t in NAMES if all(r[4] for r in rows if r[0] == t)),
    },
}, indent=1), encoding="utf-8")
print(f"\nwrote {out.relative_to(ROOT)}")
