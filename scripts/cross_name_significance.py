"""Cross-name significance tests for arm comparisons.

Eyeballing cross-name mean Sharpe is not conclusive. This module tests, over the
per-name paired deltas, whether one arm reliably beats another:

- Wilcoxon signed-rank test (non-parametric; robust to small n and non-normal
  Sharpe deltas), with a sign-test fallback when scipy is absent.
- A percentile bootstrap CI on the mean paired delta.
- The sign count (how many names favour the arm).

Caveat reported alongside: names are cross-sectionally correlated (especially
the four tech reference names), which violates the independence the paired test
assumes, so the random set is the cleaner sample. n is small; the tests are
honest but underpowered.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import fmean, median
from typing import Sequence


def _wilcoxon_p(deltas: Sequence[float]) -> tuple[float | None, str]:
    """Two-sided Wilcoxon signed-rank p-value; sign-test fallback without scipy."""
    nonzero = [d for d in deltas if d != 0.0]
    if len(nonzero) < 1:
        return None, "no nonzero deltas"
    try:
        from scipy.stats import wilcoxon
        # zero_method='wilcox' drops zeros; needs >= 1 nonzero.
        stat, p = wilcoxon(nonzero)
        return float(p), "wilcoxon-signed-rank"
    except Exception:
        # Exact two-sided sign test via the binomial.
        from math import comb
        n = len(nonzero)
        k = sum(1 for d in nonzero if d > 0)
        tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1))
        p = min(1.0, 2.0 * tail / (2 ** n))
        return float(p), "sign-test"


def _bootstrap_ci(deltas: Sequence[float], *, iters: int = 10000, seed: int = 42,
                  alpha: float = 0.05) -> tuple[float, float]:
    if not deltas:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(fmean(sample))
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return (lo, hi)


def paired_arm_test(per_name_deltas: Sequence[float]) -> dict:
    """Summary of whether the paired deltas are reliably positive."""
    deltas = [float(d) for d in per_name_deltas]
    p, method = _wilcoxon_p(deltas)
    lo, hi = _bootstrap_ci(deltas)
    return {
        "n": len(deltas),
        "mean_delta": fmean(deltas) if deltas else float("nan"),
        "median_delta": median(deltas) if deltas else float("nan"),
        "n_positive": sum(1 for d in deltas if d > 0),
        "p_value": p,
        "test": method,
        "bootstrap_ci95": [lo, hi],
        "significant_at_05": (p is not None and p < 0.05),
    }


def arm_deltas(report_dir: Path, names: Sequence[str], arm_a: str, arm_b: str,
               metric: str = "sharpe", suffix: str = "") -> list[float]:
    """Per-name (arm_a - arm_b) metric deltas, over names with both arms present.

    ``suffix`` selects a variant's reports, e.g. suffix="__det_scaled" reads
    <name>__det_scaled.json instead of <name>.json.
    """
    out: list[float] = []
    for name in names:
        path = report_dir / f"{name}{suffix}.json"
        if not path.exists():
            continue
        arms = json.loads(path.read_text(encoding="utf-8")).get("arms", {})
        a = arms.get(arm_a, {}).get("metrics", {}).get(metric)
        b = arms.get(arm_b, {}).get("metrics", {}).get(metric)
        if a is not None and b is not None:
            out.append(float(a) - float(b))
    return out


if __name__ == "__main__":
    import argparse

    REFERENCE = ("TSLA", "NFLX", "AMZN", "MSFT")
    RANDOM = ("ADSK", "BIIB", "C", "CVX", "DHI", "ECL", "PNC", "TGT")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report-dir", type=Path, default=Path("reports/final"))
    parser.add_argument("--metric", default="sharpe")
    parser.add_argument("--suffix", default="",
                        help="Variant report suffix, e.g. __det_scaled (default: main).")
    args = parser.parse_args()

    comparisons = [
        ("market_state_memory", "equal_weight", "learned trust vs equal"),
        ("market_state_memory_online", "market_state_memory", "self-improving vs frozen"),
        ("global_memory", "equal_weight", "global vs equal"),
    ]
    for group, names in (("reference", REFERENCE), ("random", RANDOM),
                         ("pooled", REFERENCE + RANDOM)):
        print(f"\n=== {group} ({args.metric}) ===")
        for a, b, label in comparisons:
            deltas = arm_deltas(args.report_dir, names, a, b, args.metric, args.suffix)
            if not deltas:
                print(f"  {label}: no data")
                continue
            r = paired_arm_test(deltas)
            star = " *" if r["significant_at_05"] else ""
            p_str = "n/a" if r["p_value"] is None else f"{r['p_value']:.3f}"
            note = "  (all deltas 0)" if r["p_value"] is None else ""
            print(f"  {label:28s} n={r['n']} mean={r['mean_delta']:+.3f} "
                  f"pos={r['n_positive']}/{r['n']} p={p_str} "
                  f"CI95=[{r['bootstrap_ci95'][0]:+.2f},{r['bootstrap_ci95'][1]:+.2f}]{star}{note}")
