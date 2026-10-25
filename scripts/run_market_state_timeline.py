from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from trading_agent.alpha.market_state_classifier import (
    MARKET_STATES,
    WARMUP,
    MarketStateClassifier,
    MarketStateConfig,
)
from trading_agent.data.finsaber_panel import load_panel
from trading_agent.market import LoaderPricePanel, _to_date


STATE_COLUMNS = (WARMUP, *MARKET_STATES)
DEFAULT_EVENTS = (
    ("2008_2009_financial_crisis", "2008-01-02", "2009-06-30"),
    ("may_2010_flash_crash", "2010-05-06", "2010-05-07"),
    ("aug_oct_2011_downgrade_eurozone_stress", "2011-08-01", "2011-10-31"),
    ("aug_2015_drawdown", "2015-08-17", "2015-08-31"),
    ("jan_2016_drawdown", "2016-01-04", "2016-01-29"),
    ("feb_2018_volatility_event", "2018-02-01", "2018-02-28"),
    ("q4_2018_volatility_event", "2018-10-01", "2018-12-31"),
    ("feb_mar_2020_crash", "2020-02-03", "2020-03-31"),
    ("2022_bear_market", "2022-01-03", "2022-12-30"),
    ("mar_2023_regional_bank_stress", "2023-03-08", "2023-03-31"),
    ("aug_2024_volatility_spike", "2024-08-01", "2024-08-31"),
)


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _read_protocol(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _counts_frame(labels: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for value, group in labels.groupby(key):
        counts = Counter(group["state"])
        row = {key: value}
        row.update({state: int(counts.get(state, 0)) for state in STATE_COLUMNS})
        row["total"] = int(len(group))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def _window_counts(labels: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    rows = []
    for window in windows:
        start = _parse_date(window["start"])
        end = _parse_date(window["end"])
        subset = labels[(labels["date"] >= start) & (labels["date"] <= end)]
        counts = Counter(subset["state"])
        row = {
            "name": window["name"],
            "start": window["start"],
            "end": window["end"],
        }
        row.update({state: int(counts.get(state, 0)) for state in STATE_COLUMNS})
        row["total"] = int(len(subset))
        rows.append(row)
    return pd.DataFrame(rows)


def _episode_frame(labels: pd.DataFrame, states: set[str]) -> pd.DataFrame:
    rows = []
    active = labels[labels["state"].isin(states)].reset_index(drop=True)
    if active.empty:
        return pd.DataFrame(columns=["state", "start", "end", "trading_days"])

    start = active.loc[0, "date"]
    end = start
    state = active.loc[0, "state"]
    n = 1
    prev_pos = int(active.loc[0, "_pos"])
    for _, row in active.iloc[1:].iterrows():
        pos = int(row["_pos"])
        row_state = str(row["state"])
        if pos == prev_pos + 1 and row_state == state:
            end = row["date"]
            n += 1
        else:
            rows.append({"state": state, "start": start.isoformat(), "end": end.isoformat(), "trading_days": n})
            start = row["date"]
            end = start
            state = row_state
            n = 1
        prev_pos = pos
    rows.append({"state": state, "start": start.isoformat(), "end": end.isoformat(), "trading_days": n})
    return pd.DataFrame(rows)


def _event_cross_check(labels: pd.DataFrame, adjacent_trading_days: int) -> pd.DataFrame:
    rows = []
    for name, start_raw, end_raw in DEFAULT_EVENTS:
        start = _parse_date(start_raw)
        end = _parse_date(end_raw)
        in_window = labels[(labels["date"] >= start) & (labels["date"] <= end)]
        if in_window.empty:
            lo_pos = labels[labels["date"] < start]["_pos"].max()
            hi_pos = labels[labels["date"] > end]["_pos"].min()
        else:
            lo_pos = int(in_window["_pos"].min())
            hi_pos = int(in_window["_pos"].max())
        if pd.isna(lo_pos) or pd.isna(hi_pos):
            adjacent = in_window
        else:
            adjacent = labels[
                (labels["_pos"] >= int(lo_pos) - adjacent_trading_days)
                & (labels["_pos"] <= int(hi_pos) + adjacent_trading_days)
            ]
        hit = adjacent[adjacent["state"].isin({"elevated", "crisis"})]
        counts = Counter(adjacent["state"])
        if hit.empty:
            first = ""
            last = ""
            finding = "no elevated or crisis label within or adjacent to event window"
        else:
            first = hit["date"].min().isoformat()
            last = hit["date"].max().isoformat()
            finding = "elevated/crisis label appears within or adjacent to event window"
        rows.append({
            "event": name,
            "event_start": start_raw,
            "event_end": end_raw,
            "adjacent_trading_days": adjacent_trading_days,
            "elevated": int(counts.get("elevated", 0)),
            "crisis": int(counts.get("crisis", 0)),
            "first_elevated_or_crisis": first,
            "last_elevated_or_crisis": last,
            "finding": finding,
        })
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    *,
    manifest: dict,
    yearly: pd.DataFrame,
    reserved: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    diagnostic_match: dict,
) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        columns = [str(col) for col in frame.columns]
        rows = []
        rows.append("| " + " | ".join(columns) + " |")
        rows.append("| " + " | ".join("---" for _ in columns) + " |")
        for _, item in frame.iterrows():
            rows.append("| " + " | ".join(str(item[col]) for col in frame.columns) + " |")
        return "\n".join(rows)

    lines = [
        "# Full-Span Market-State Timeline",
        "",
        "No trading, expert calls, LLM calls, thresholds, weights, or performance figures are included.",
        "",
        "## Manifest",
        f"- data_root: {manifest['data_root']}",
        f"- active_start: {manifest['active_start']}",
        f"- active_end: {manifest['active_end']}",
        f"- preload_start: {manifest['preload_start']}",
        f"- tickers: {', '.join(manifest['tickers'])}",
        f"- classifier: {manifest['classifier']}",
        f"- classifier_config: {json.dumps(manifest['classifier_config'], sort_keys=True)}",
        f"- diagnostic_2022_2023_match: {diagnostic_match['match']}",
        "",
        "## Per-Year Counts",
        md_table(yearly),
        "",
        "## Reserved Test Period Counts",
        md_table(reserved),
        "",
        "## Elevated/Crisis Episodes",
        md_table(episodes),
        "",
        "## Event Cross-Check",
        md_table(events),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _try_plot(labels: pd.DataFrame, output: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    code = {WARMUP: 0, "up_calm": 1, "up_stress": 2, "down_calm": 3, "down_stress": 4}
    fig, ax = plt.subplots(figsize=(14, 3))
    ax.step(labels["date"], labels["state"].map(code), where="post", linewidth=0.7)
    ax.set_yticks(list(code.values()), list(code.keys()))
    ax.set_title("Market-state labels")
    ax.set_xlabel("date")
    ax.set_ylabel("state")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return str(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write full-span market-state label artifacts.")
    parser.add_argument("--data-root", default="data/finsaber/sp500_2000_2025_parquet")
    parser.add_argument("--protocol", default="experiment_protocols/main_walk_forward.json")
    parser.add_argument("--active-start", type=_parse_date, default=date(2004, 1, 2))
    parser.add_argument("--active-end", type=_parse_date, default=date(2025, 12, 31))
    parser.add_argument("--preload-start", type=_parse_date, default=date(2000, 1, 3))
    parser.add_argument("--output-dir", default="reports/market_state_timeline_2004_2025")
    parser.add_argument("--adjacent-trading-days", type=int, default=5)
    args = parser.parse_args(argv)

    protocol = _read_protocol(Path(args.protocol))
    tickers = list(protocol["data_scope"]["universe"])
    cfg = MarketStateConfig()
    loader = load_panel(args.data_root, args.preload_start, args.active_end, tickers=tickers, modalities=("price",))
    panel = LoaderPricePanel(loader)
    classifier = MarketStateClassifier(panel, cfg)

    rows = []
    active_dates = [d for d in panel.dates if args.active_start <= _to_date(d) <= args.active_end]
    for pos, dt in enumerate(active_dates):
        state = classifier.classify_at(dt, tickers)
        rows.append({"date": _to_date(dt), "state": state, "_pos": pos})
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise RuntimeError("no active labels produced")

    yearly = _counts_frame(labels.assign(year=labels["date"].map(lambda d: d.year)), "year")
    reserved = _window_counts(labels, protocol["reserved_test_periods"])
    episodes = _episode_frame(labels, {"elevated", "crisis"})
    events = _event_cross_check(labels, args.adjacent_trading_days)
    diagnostic_counts = {"up_calm": 85, "up_stress": 137, "down_calm": 113, "down_stress": 166}
    diagnostic_window = labels[
        (labels["date"] >= date(2022, 1, 3))
        & (labels["date"] <= date(2023, 12, 29))
    ]
    observed_diagnostic_counts = {
        state: int((diagnostic_window["state"] == state).sum())
        for state in MARKET_STATES
    }
    diagnostic_match = {
        "expected": diagnostic_counts,
        "observed": observed_diagnostic_counts,
        "match": observed_diagnostic_counts == diagnostic_counts,
    }
    if not diagnostic_match["match"]:
        raise RuntimeError(f"2022-2023 diagnostic counts mismatch: {diagnostic_match}")

    if (labels["state"] == WARMUP).any():
        warmup_after_start = labels[labels["state"] == WARMUP]
        raise RuntimeError(
            "warmup labels remain after active start: "
            f"{warmup_after_start['date'].min()}..{warmup_after_start['date'].max()}"
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels[["date", "state"]].to_csv(out / "daily_market_state_labels.csv", index=False)
    yearly.to_csv(out / "per_calendar_year_counts.csv", index=False)
    reserved.to_csv(out / "reserved_test_period_counts.csv", index=False)
    episodes.to_csv(out / "elevated_crisis_episodes.csv", index=False)
    events.to_csv(out / "event_cross_check.csv", index=False)
    manifest = {
        "data_root": args.data_root,
        "active_start": args.active_start.isoformat(),
        "active_end": args.active_end.isoformat(),
        "preload_start": args.preload_start.isoformat(),
        "tickers": tickers,
        "classifier": MarketStateClassifier.semantics_version,
        "classifier_config": asdict(cfg),
        "n_active_trading_days": int(len(labels)),
        "state_columns": list(STATE_COLUMNS),
        "reserved_test_periods": protocol["reserved_test_periods"],
        "diagnostic_2022_2023_match": diagnostic_match,
        "outputs": {
            "labels_csv": str(out / "daily_market_state_labels.csv"),
            "yearly_counts_csv": str(out / "per_calendar_year_counts.csv"),
            "reserved_counts_csv": str(out / "reserved_test_period_counts.csv"),
            "episodes_csv": str(out / "elevated_crisis_episodes.csv"),
            "event_cross_check_csv": str(out / "event_cross_check.csv"),
        },
    }
    plot_path = _try_plot(labels, out / "label_series.png")
    if plot_path:
        manifest["outputs"]["plot"] = plot_path
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(
        out / "market_state_timeline_report.md",
        manifest=manifest,
        yearly=yearly,
        reserved=reserved,
        episodes=episodes,
        events=events,
        diagnostic_match=diagnostic_match,
    )
    print(json.dumps(manifest["outputs"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
