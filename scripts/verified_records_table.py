"""Write the appendix table of every verified record.

Section 5.1 names which stocks carry records and says that AES and CTAS also carry state
records, but it never gives their counts, so a reader adding up that paragraph reaches
fourteen while the text elsewhere says nineteen. This table is the missing enumeration:
one row per verified (memory, market state, expert) cell, with the peer-relative mean the
test measured, its sample size and the Holm-adjusted p that passed it.

    TRADING_AGENT_DATA_ROOT=... PYTHONPATH=src python scripts/verified_records_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from portfolio20 import resolve  # noqa: E402

ARMS = (("global_memory", "global"), ("market_state_memory", "state"))
# The record column carries the state name, or "global" for the state-free memory.
STATE_NAME = {
    "global": "global",
    "up_calm": "Up-Calm",
    "up_stress": "Up-Stress",
    "down_calm": "Down-Calm",
    "down_stress": "Down-Stress",
}


def rows() -> list[tuple]:
    sealed, _ = resolve()
    out = []
    for ticker in sorted(sealed):
        report = json.loads(sealed[ticker].read_text(encoding="utf-8"))
        checks = report.get("verify_gate_report") or {}
        for arm_key, arm_label in ARMS:
            per_cell = (checks.get(arm_key) or {}).get("per_cell") or {}
            for cell, stats in per_cell.items():
                if not stats.get("verified"):
                    continue
                state, expert = cell.split("|", 1)
                out.append((
                    ticker,
                    arm_label,
                    STATE_NAME.get(state, state),
                    expert,
                    float(stats["mean_m"]),
                    int(stats["n"]),
                    float(stats["p_holm"]),
                ))
    # global before state, then by stock, then by expert
    out.sort(key=lambda r: (r[1] != "global", r[0], r[3]))
    return out


def main() -> int:
    data = rows()
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"\textbf{Stock} & \textbf{Record} & \textbf{Expert} & "
        r"\textbf{Mean} & \textbf{$n$} & \textbf{$p_{\mathrm{Holm}}$} \\",
        r"\midrule",
    ]
    prev_arm = None
    for ticker, arm, state, expert, mean, n, p in data:
        if prev_arm is not None and arm != prev_arm:
            lines.append(r"\midrule")
        prev_arm = arm
        lines.append(
            f"{ticker} & {state} & {expert} & "
            f"${mean:+.4f}$ & ${n:,}$ & ${p:.3f}$ \\\\".replace(",", "{,}")
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Every record that passed verification, the nineteen the freeze wrote "
        r"down. Mean is the peer-relative contribution over the training window, $n$ the "
        r"episodes behind it, $p_{\mathrm{Holm}}$ the corrected $p$; the records below "
        r"the rule are read only in the state they name.}",
        r"\label{tab:verified-records}",
        r"\end{table}",
    ]
    out = ROOT / "reports" / "final" / "verified_records_table.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(data)} verified records -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
