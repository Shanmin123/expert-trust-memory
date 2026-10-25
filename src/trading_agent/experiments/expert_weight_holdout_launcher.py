from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

from trading_agent.alpha.market_state_classifier import MarketStateConfig
from trading_agent.alpha.alpha158_strategy import available_tickers_from_sqlite_qlib_factors
from trading_agent.contracts import BacktestConfig
from trading_agent.data.finsaber_panel import load_panel, resolve_data_root, select_fixed_universe
from trading_agent.experts.backends import (
    DEFAULT_CODEX_SDK_MODEL,
    DEFAULT_EXPERT_MODEL,
    SUPPORTED_EXPERT_BACKENDS,
    build_experts,
    expert_backend_metadata,
)
from trading_agent.experts.backends import OpenAIChatBackend
from trading_agent.experts.view_corpus import (
    READTHROUGH,
    RECORD,
    REPLAY,
    REPLAY_BY_DATE,
    ViewCorpus,
    attach_corpus,
)
from trading_agent.experiments.expert_weight_holdout import (
    ExpertWeightHoldoutResult,
    _arm_comparison_vs_equal_weight,
    _metrics,
    run_expert_weight_holdout,
)
from trading_agent.expert_context import build_expert_signal_sources
from trading_agent.experiments.expert_reliability_diagnostic import (
    buy_hold_sell_summary,
    expert_reliability_report,
    view_direction_summary,
)
from trading_agent.memory.expert_scorecard import ExpertScorecardStore
from trading_agent.memory.expert_weights import MARKET_STATES
from trading_agent.reporting import REPORT_SCHEMA_VERSION, SYSTEM_IDENTITY
from trading_agent.runner import run_expert_loop


@dataclass(frozen=True)
class _UniverseSelection:
    tickers: tuple[str, ...]
    label: str


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_ticker_list(value: str) -> tuple[str, ...]:
    out = tuple(part.strip().upper() for part in str(value).split(",") if part.strip())
    if not out:
        raise argparse.ArgumentTypeError("at least one ticker is required")
    return out


def _parse_exit_threshold(value: str):
    raw = str(value).strip().lower()
    if raw in {"none", "off"}:
        return None
    return float(value)


def _parse_thresholds(value: str) -> tuple[float, ...]:
    out = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        threshold = float(raw)
        # The aggregate score A = sum w*v lies in [-1, 1], so a NEGATIVE entry
        # threshold is a valid participation-biased rule: enter long unless the
        # committee is strongly bearish. Allow [-1, 1].
        if not -1.0 <= threshold <= 1.0:
            raise argparse.ArgumentTypeError("decision thresholds must be in [-1, 1]")
        out.append(threshold)
    if not out:
        raise argparse.ArgumentTypeError("at least one decision threshold is required")
    return tuple(out)


def _parse_temperatures(value: str) -> tuple[float, ...]:
    out = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        temperature = float(raw)
        if temperature <= 0.0:
            raise argparse.ArgumentTypeError("softmax temperatures must be positive")
        out.append(temperature)
    if not out:
        raise argparse.ArgumentTypeError("at least one temperature is required")
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the expert-weight holdout experiment.")
    parser.add_argument("--data-root", default=None, help="FINSABER parquet root; defaults to FINSABER_DATA_ROOT.")
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--validation-boundary", type=_parse_date, default=None)
    parser.add_argument("--boundary", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--names", type=int, default=10)
    parser.add_argument(
        "--tickers", type=_parse_ticker_list, default=None,
        help="Comma-separated explicit ticker universe (e.g. TSLA for a single-name run). "
        "Overrides --names and any protocol universe.",
    )
    parser.add_argument("--embargo-days", type=int, default=5)
    parser.add_argument("--reflect-every", type=int, default=21)
    parser.add_argument("--reflection-window", type=int, default=63)
    parser.add_argument("--reflection-min-support", type=int, default=5)
    parser.add_argument(
        "--terminal-verify", action="store_true",
        help="opt-in TERMINAL verification (backend-agnostic, applies to LLM and deterministic "
        "makers alike): ONE pass at freeze time clears every reflection delta whose "
        "(market_state, expert) cell lacks statistically verified marginal skill "
        "(m-proxy, synchronized-date studentized block bootstrap, Holm FWER). "
        "The verification-on/off ablation. Grid-invariant; fixed a-priori params.",
    )
    parser.add_argument(
        "--verify-alpha", type=float, default=0.10,
        help="Significance level for the terminal verification (default .10). Higher = looser "
        "verification (more cells pass). Diagnostic knob for the strict-vs-loose calibration "
        "study; not the sealed default.",
    )
    parser.add_argument(
        "--verify-correction", choices=("holm", "fdr"), default="holm",
        help="Multiple-comparison control for the terminal verification (with --terminal-verify): "
        "holm (default, familywise error) or fdr (Benjamini-Hochberg false discovery rate, less "
        "conservative so more cells can act). Redesign knob, not the sealed default.",
    )
    parser.add_argument(
        "--verify-persistence", action="store_true",
        help="Out-of-sample-generalisation filter (with --terminal-verify): a verified cell must "
        "also keep the same-signed mean contribution in BOTH halves of the training window "
        "(temporal persistence), rejecting in-sample flukes. Opt-in, not the sealed default.",
    )
    parser.add_argument(
        "--verify-min-effect", type=float, default=0.0,
        help="Effect-size floor (with --terminal-verify): a verified cell must have "
        "|mean marginal contribution| >= this value, rejecting tiny-but-significant noise. "
        "0 disables (sealed default).",
    )
    parser.add_argument(
        "--verify-block", type=int, default=None,
        help="Moving-block bootstrap block length for terminal verification. Default is "
        "max(2*h, 14); h-day overlapping returns lose their autocorrelation at lag h, so h "
        "itself is the minimum that preserves the dependence and 2h is extra conservatism "
        "(prereg Amendment 9).",
    )
    parser.add_argument(
        "--verify-kfold", type=int, default=0,
        help="Temporal k-fold confirmation (with --terminal-verify): a verified cell must keep "
        "the sign of its mean marginal contribution in >= k-1 of k contiguous folds of its own "
        "training episodes (full-train replacement for thin held-out-window confirmation). "
        "0 disables (sealed default); pre-registered config is 5.",
    )
    parser.add_argument(
        "--verify-oos", action="store_true",
        help="Out-of-sample confirmation (with --terminal-verify): replay the held-out "
        "validation window and require each verified cell to keep its sign there, dropping "
        "in-sample-only false positives. Opt-in, not the sealed default.",
    )
    parser.add_argument(
        "--reflection-backend", choices=("deterministic", "deterministic_scaled", "llm"),
        default="deterministic",
        help="llm = LLMExpertReflection over the chat API (deterministic rule stays the "
        "fallback); deterministic = the symmetric fixed-scale rule; deterministic_scaled "
        "= the scale-free rule (best/worst experts get near-bound deltas).",
    )
    parser.add_argument(
        "--reflection-model", default=None,
        help="Chat model for --reflection-backend llm; defaults to --expert-model.",
    )
    parser.add_argument(
        "--dsr-trial-count", type=int, default=None,
        help="Override (enlarge only) the deflated-Sharpe trial budget; the "
             "pre-registered free-replay families report at the cumulative 18.")
    parser.add_argument(
        "--reflection-step-mode", choices=("fixed", "scale_free", "t_stat"),
        default="scale_free",
        help="Deterministic reflection step rule. t_stat is the pre-registered "
             "uncertainty-scaled variant.")
    parser.add_argument(
        "--memory-half-life", type=float, default=None,
        help="EW decay half-life override (pre-registered robustness replay; "
             "default keeps the a-priori 63).")
    parser.add_argument(
        "--variance-penalty", type=float, default=None,
        help="Retrieval variance penalty lambda override (pre-registered "
             "robustness replay; default keeps the a-priori 1.0).")
    parser.add_argument(
        "--memory-score", choices=("ewvar", "hit_logodds"), default="ewvar",
        help="Retrieval score (Family E1): 'ewvar' = sealed mean-variance score; "
             "'hit_logodds' = Laplace-shrunk hit-rate log-odds.")
    parser.add_argument(
        "--stale-view-abstain", type=int, default=0,
        help="Family F: an expert whose direction has repeated for more than N active "
             "dates leaves the vote until it changes (0 = off, the sealed default). "
             "Learning evidence is unaffected.")
    parser.add_argument(
        "--memory-shrinkage", choices=("none", "empirical_bayes"), default="none",
        help="Family S: 'empirical_bayes' pulls each state cell toward the global card by "
             "its own reliability B = tau^2/(tau^2+se^2) instead of a binary certify/reject "
             "decision. Replaces --certified-only-memory rather than composing with it.")
    parser.add_argument(
        "--position-sizing", choices=("binary", "proportional"), default="binary",
        help="Decision layer (Family E2): 'binary' = sealed full-invest/flat; "
             "'proportional' = target fraction clip(A,0,1) while long, "
             "entry/exit dates unchanged.")
    parser.add_argument(
        "--standardized-memory", action="store_true", default=True,
        help="Dimensionless retrieval (system default): z = (q / pooled EW std) / tau, so "
        "tau is a pure number (grid 0.5,1,2) and reflection deltas are commensurate with z.",
    )
    parser.add_argument(
        "--raw-memory", dest="standardized_memory", action="store_false",
        help="Ablation: raw q/tau retrieval (the original dimensioned score); pair with a "
        "small tau grid like 0.0005,0.001,0.002.",
    )
    parser.add_argument(
        "--shrink-to-global", action="store_true",
        help="Hierarchical shrinkage for the state arm: under-supported states blend "
        "toward the pooled (global) moments with lambda = n/(n+min_support) instead of "
        "falling back to equal weights.",
    )
    parser.add_argument(
        "--numeric-only-memory", action="store_true",
        help="Ablation: retrieval ignores the reflection deltas (use_reflection=False); "
        "the scorecard moments are unchanged.",
    )
    parser.add_argument(
        "--reflection-under-certified", action="store_true",
        help="Deprecated no-op (2026-07-21): certified-only retrieval keeps the bounded "
        "reflection delta by default, matching the sealed canonical runs. The prior "
        "'provably inert' rationale held only at the saturated snapshot tau=0.001, not "
        "the selected tau=0.5 (BIIB max weight shift 0.041), and is retracted. Use "
        "--numeric-only-memory for the explicit delta-off ablation.",
    )
    parser.add_argument(
        "--certified-only-memory", action="store_true",
        help="Never-harm retrieval (2026-07-19): weights may deviate from equal "
        "ONLY in cells the terminal verification certified; states with nothing "
        "certified trade equal-weight. Pre-specified secondary for the sealed "
        "protocol.",
    )
    parser.add_argument(
        "--participation-prior", action="store_true",
        help="Append the upward-drift base rate to each expert prompt to counter the "
        "flat-bias (opt-in experiment; changes the corpus cache key, so it needs its "
        "own recorded corpus).",
    )
    parser.add_argument(
        "--parallel-experts", action="store_true",
        help="Call the experts concurrently per date (the view corpus is "
        "thread-safe); ~4x faster with a live LLM backend.",
    )
    # theta FROZEN at 0 a priori (2026-07-17, round-2 review concurrence): the
    # validation window is exactly flat across the old {0.0, 0.25, 0.5} grid --
    # theta is unidentifiable -- so committee-majority semantics (enter on
    # strict net support) is fixed instead of pretending validation tunes it.
    # The tuned grid is now temperature-only (trial_count = len(tau_grid)).
    # Passing an explicit multi-point grid remains available as a DIAGNOSTIC.
    # (History: the pre-2026-07-17 default was {0.0, 0.25, 0.5}; the older
    # {0.3,0.4,0.5} was degenerate -- all three required 3+ longs of 4, and 0.5
    # sat on the 3-vs-1 vote knife-edge.)
    parser.add_argument("--decision-thresholds", type=_parse_thresholds, default=(0.0,))
    parser.add_argument(
        "--exit-threshold", type=_parse_exit_threshold, default=0.0,
        help="HOLD semantics (system default: 0): enter long at score STRICTLY "
        "above the decision threshold, then HOLD while the score stays at or "
        "above this value; exit strictly below it (0 = exit when the committee "
        "net vote turns negative; a zero net vote maintains the position). "
        "Fixed a priori, NOT part of the tuned grid. Pass 'none' to disable "
        "and re-vote the entry threshold every day.",
    )
    parser.add_argument(
        "--decision-cadence", type=int, default=1,
        help="Re-vote the committee only every Nth processed date (positions carry between). "
        "1 = daily (sealed default). Coarser decision horizon; learning "
        "evidence is unchanged (attribution is action-independent).",
    )
    parser.add_argument(
        "--soft-abstain-weight", type=float, default=0.0,
        help="Soft abstention (opt-in): a deliberate neutral view's forced_direction joins the "
        "committee at this fraction of a vote (a-priori 0.5 suggested), but only when at least "
        "one full speaker exists (silence still maintains). 0 disables (sealed default).",
    )
    parser.add_argument(
        "--confidence-weighted-vote", action="store_true", dest="confidence_weighted_vote",
        help="Scale each expert's vote by its stated confidence in the committee aggregation "
        "(uses the otherwise-discarded confidence channel; changes only the decision, not the "
        "learned memory). Opt-in experiment, not the sealed default.",
    )
    parser.add_argument(
        "--vote-policy", choices=("ternary", "binary"), default="ternary",
        help="Replay policy for the D3 comparison: 'ternary' (system default) "
        "keeps deliberate neutrals as abstentions; 'binary' maps each neutral "
        "onto the forced_direction captured in the SAME response "
        "(design-D-style binary committee over the same corpus).",
    )
    parser.add_argument(
        "--temperature-grid", type=_parse_temperatures, default=None,
        help="Softmax temperatures swept on validation (e.g. 0.0005,0.001,0.002 for the "
        "three-point grid). Default: a single temperature (scorecard default), i.e. no "
        "temperature tuning. Each temperature is a numeric re-read of the frozen scorecard.",
    )
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--commission-per-share", type=float, default=0.0049)
    parser.add_argument("--min-commission", type=float, default=0.99)
    parser.add_argument("--max-commission-rate", type=float, default=0.01)
    parser.add_argument("--slippage-perc", type=float, default=0.0005)
    parser.add_argument("--liquidity-cap-pct", type=float, default=0.0)
    parser.add_argument("--headline-cost-bps-per-side", type=float, default=10.0)
    parser.add_argument("--sensitivity-cost-bps-per-side", type=float, default=25.0)
    parser.add_argument("--preload-calendar-days", type=int, default=900)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vol-window", type=int, default=21)
    parser.add_argument("--percentile-window", type=int, default=504)
    parser.add_argument("--trend-window", type=int, default=200,
                        help="Trailing SMA window for the trend dimension of market_state.")
    parser.add_argument("--trend-confirm-days", type=int, default=5,
                        help="Consecutive days on the new side required to confirm a trend switch.")
    parser.add_argument("--vol-quartile-regime", action="store_true",
                        help="Robustness variant: use the volatility-only four-state label "
                        "(Harvey 2019 style) instead of the 2D trend x risk label, to test "
                        "whether the trend axis matters.")
    parser.add_argument("--vol-tercile-regime", action="store_true",
                        help="Family Z1: three balanced volatility bins (Amendment 13).")
    parser.add_argument("--trend-cond-risk-regime", action="store_true",
                        help="Family Z2: trend x risk with the risk median computed within "
                             "the trend leg, balancing the 2D label by construction.")
    parser.add_argument("--down-merged-regime", action="store_true",
                        help="Family Z3: the 2D label with down_calm merged into down_stress.")
    parser.add_argument("--trend-only-regime", action="store_true",
                        help="Coarsening variant: use the two-state up/down label (drops the "
                        "risk axis) instead of the 2D trend x risk label, to test whether the "
                        "state arm's inertness is a per-cell support problem. Mutually exclusive "
                        "with --vol-quartile-regime.")
    parser.add_argument("--risk-binary-regime", action="store_true",
                        help="Coarsening variant: use the two-state calm/stress label (drops the "
                        "trend axis, coarsest volatility partition). Mutually exclusive with the "
                        "other regime flags.")
    parser.add_argument("--alpha158-factors-db", default=None)
    parser.add_argument("--qlib-provider-uri", default=None)
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--no-filings", action="store_true",
                        help="Disable the fundamental expert's 10-K/10-Q filing signals "
                             "(no-filings ablation); filings are used by default when present.")
    parser.add_argument("--strict-news", action="store_true")
    parser.add_argument("--sec-facts-path", default=None,
                        help="Parquet of SEC XBRL financials (scripts/collect_sec_xbrl.py). When "
                             "given, the fundamental expert reads point-in-time annual ratios "
                             "(margins/ROA/leverage/FCF) -- the price-orthogonal fundamental core.")
    parser.add_argument(
        "--prior-lessons-path", default=None,
        help="E-SKILL fed arm: JSON file of frozen train-window text lessons "
        "(flat list or the report's lessons_by_state mapping) fed to every "
        "expert as a constant prior_lessons block. Changes prompts -> fresh "
        "corpus required for the fed arm.",
    )
    parser.add_argument("--fred-macro-path", default=None,
                        help="Parquet of FRED macro series (scripts/collect_fred_macro.py). When "
                             "given, the news expert's context carries a publication-lag-embargoed "
                             "macro block (rates/curve/CPI/claims), per the TradingAgents news-"
                             "analyst design.")
    parser.add_argument("--expert-backend", choices=SUPPORTED_EXPERT_BACKENDS, default="static")
    parser.add_argument("--expert-model", default=None)
    parser.add_argument("--expert-model-revision", default=None)
    parser.add_argument("--news-sentiment-cache", default=None)
    parser.add_argument("--news-sentiment-model", default="gpt-4.1-mini")
    parser.add_argument("--protocol", default="experiment_protocols/main_walk_forward.json")
    parser.add_argument("--run-label", default="HOLDOUT")
    parser.add_argument("--launcher", default="run_expert_weight_holdout.py")
    parser.add_argument("--output", default="reports/expert_weight_holdout")
    parser.add_argument("--memory-output", default=None)
    parser.add_argument(
        "--export-daily-views", default=None,
        help="Optional JSONL path: write the equal arm's per-date expert views "
        "(direction, confidence, stated reasoning, market_state) for the test "
        "window. Audit/export only; nothing in the decision path reads it.",
    )
    parser.add_argument(
        "--view-corpus",
        default=None,
        help="Path to a frozen expert-view corpus (JSONL). Record real LLM views once "
        "and replay them deterministically across the holdout's internal runs.",
    )
    parser.add_argument(
        "--unbounded-memory",
        action="store_true",
        help="Disable the cumulative per-key delta clip (E3 unbounded-memory ablation). "
        "The per-entry +/-0.10 clip still applies; only the summed-delta cap is removed.",
    )
    parser.add_argument(
        "--market-proxy",
        default=None,
        help="Path to an external market-proxy parquet (date, adjusted_close), e.g. "
        "data/market_proxy/spy_daily.parquet. Required for single-name runs: the "
        "market-state label and the attribution baseline use this series instead of "
        "the equal-weight universe (which would be the traded stock itself).",
    )
    parser.add_argument(
        "--attribution-mode",
        choices=("raw", "market_adjusted"),
        default="raw",
        help="raw (default): reflection scores each expert on its counterfactual net "
        "contribution over the raw per-name return, not demeaned; market_adjusted: demeans "
        "against the market proxy before scoring (E2 ablation).",
    )
    parser.add_argument(
        "--attribution-horizon",
        type=int,
        default=1,
        help="Learning horizon h (active bars) for the counterfactual g = v * r_{t->t+h}. "
        "Default 1 (next-bar). The horizon study found expert directional edge is strongest "
        "at h=21 (monthly), masked at daily; h>1 scores learning at that horizon and the loop "
        "embargoes each episode by h bars so no decision reads its own future (leak-safe).",
    )
    parser.add_argument(
        "--view-corpus-mode",
        choices=(READTHROUGH, REPLAY, REPLAY_BY_DATE, RECORD),
        default=READTHROUGH,
        help="readthrough: serve cached views, call the backend only on a miss and record it "
        "(the right mode for recording a holdout corpus, since arms/threshold sweeps reuse "
        "identical prompts); replay: cached only by exact prompt hash, error on a miss "
        "(strict reproduction); replay_by_date: like replay but falls back to the frozen view "
        "for the exact (expert, date, tickers) request when serialisation drift moves the prompt "
        "hash (for downstream experiments that keep the views fixed); record: always call the "
        "backend and append.",
    )
    return parser


def _config(args: argparse.Namespace, tickers: Sequence[str]) -> BacktestConfig:
    # A positive bps cost replaces the per-share commission model inside the
    # execution adapter, so the commission parameters would be silently ignored.
    # Zero them here so the config fingerprint and manifest state the cost model
    # that is actually charged.
    bps_model_active = float(args.headline_cost_bps_per_side) > 0
    return BacktestConfig(
        date_from=args.start,
        date_to=args.end,
        preload_date_from=args.preload_start,
        initial_cash=args.initial_cash,
        tickers=tuple(tickers),
        commission_per_share=0.0 if bps_model_active else args.commission_per_share,
        min_commission=0.0 if bps_model_active else args.min_commission,
        max_commission_rate=args.max_commission_rate,
        slippage_perc=args.slippage_perc,
        execution_cost_bps_per_side=args.headline_cost_bps_per_side,
        liquidity_cap_pct=args.liquidity_cap_pct,
        seed=args.seed,
    )


def _preload_start(start: date, calendar_days: int) -> date:
    return start - timedelta(days=max(0, int(calendar_days)))


def _jsonable(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _episode_counts_by_state_action(loop_result) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for episode in loop_result.episodes:
        state = episode.market_state
        action = getattr(episode, "action", "unknown")
        out.setdefault(state, {"long": 0, "flat": 0})
        out[state][action] = out[state].get(action, 0) + 1
    return out


def _pre_refactor_episode_counts(loop_result) -> dict[str, int]:
    """Universe-level proxy: one episode per active date/state before per-name refactor."""
    counts: dict[str, int] = {}
    seen_dates: set[tuple[str, str]] = set()
    for episode in loop_result.episodes:
        key = (episode.decision_date.isoformat(), episode.market_state)
        if key in seen_dates:
            continue
        seen_dates.add(key)
        counts[episode.market_state] = counts.get(episode.market_state, 0) + 1
    return counts


def _episode_count_comparison(loop_result) -> dict[str, dict[str, int]]:
    before = _pre_refactor_episode_counts(loop_result)
    after = {
        state: sum(actions.values())
        for state, actions in _episode_counts_by_state_action(loop_result).items()
    }
    states = sorted(set(before) | set(after))
    return {
        state: {
            "before_universe_level": int(before.get(state, 0)),
            "after_per_name": int(after.get(state, 0)),
            "increase": int(after.get(state, 0) - before.get(state, 0)),
        }
        for state in states
    }


def _market_state_labels(loop_result) -> list[dict[str, str]]:
    return [
        {"date": _to_iso_date(day), "state": str(state)}
        for day, state in getattr(loop_result, "market_states", [])
    ]


def _state_power(loop_result, *, min_support: int) -> dict[str, dict[str, object]]:
    counts = _episode_counts_by_state_action(loop_result)
    out: dict[str, dict[str, object]] = {}
    for state in MARKET_STATES:
        actions = counts.get(state, {"long": 0, "flat": 0})
        long_count = int(actions.get("long", 0))
        episode_count = int(sum(actions.values()))
        out[state] = {
            "episode_count": episode_count,
            "long_count": long_count,
            "flat_count": int(actions.get("flat", 0)),
            "min_support": int(min_support),
            "absent": episode_count == 0,
            "underpowered": long_count < int(min_support),
            "exclude_from_headline_comparison": long_count < int(min_support),
        }
    return out


def _state_power_by_arm(result: ExpertWeightHoldoutResult, *, min_support: int) -> dict[str, dict[str, dict[str, object]]]:
    return {
        name: _state_power(arm.loop_result, min_support=min_support)
        for name, arm in result.arms.items()
    }


def _per_state_arm_comparison_vs_equal_weight(
    result: ExpertWeightHoldoutResult,
    *,
    min_support: int,
) -> dict[str, dict[str, dict[str, object]]]:
    baseline = result.per_state_arm_metrics.get("equal_weight", {})
    power_by_arm = _state_power_by_arm(result, min_support=min_support)
    out: dict[str, dict[str, dict[str, object]]] = {}
    for arm_name, states in result.per_state_arm_metrics.items():
        if arm_name == "equal_weight":
            continue
        out[arm_name] = {}
        for state in sorted(set(states) | set(power_by_arm.get(arm_name, {}))):
            metrics = states.get(state, {})
            base = baseline.get(state, {})
            power = power_by_arm.get(arm_name, {}).get(state, {})
            out[arm_name][state] = {
                "mean_market_return_delta": float(
                    metrics.get("mean_market_return", 0.0) - base.get("mean_market_return", 0.0)
                ),
                "mean_signal_component_delta": float(
                    metrics.get("mean_signal_component", 0.0) - base.get("mean_signal_component", 0.0)
                ),
                "participation_rate_delta": float(
                    metrics.get("participation_rate", 0.0) - base.get("participation_rate", 0.0)
                ),
                **power,
            }
    return out


def _underpowered_cells(result: ExpertWeightHoldoutResult, *, min_support: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arm_name, states in _state_power_by_arm(result, min_support=min_support).items():
        for state, item in states.items():
            if item.get("underpowered"):
                rows.append({"arm": arm_name, "state": state, **item})
    return rows


def _read_protocol(path: str | Path) -> dict:
    protocol_path = Path(path)
    if not protocol_path.exists():
        return {"reserved_test_periods": []}
    return json.loads(protocol_path.read_text(encoding="utf-8"))


def _protocol_universe_has_price_data(
    data_root: str,
    tickers: Sequence[str],
    *,
    start: date,
    end: date,
) -> bool:
    try:
        panel = load_panel(data_root, start, end, tickers=tickers, modalities=("price",))
        frame = panel.get_price_dataframe(tickers=tickers, date_from=start, date_to=end, adjust=True)
    except Exception:
        return False
    if frame.empty or "symbol" not in frame:
        return False
    return bool(set(frame["symbol"].astype(str)) & {str(ticker) for ticker in tickers})


def _resolve_universe_selection(
    data_root: str,
    *,
    start: date,
    end: date,
    names: int,
    protocol: dict,
    allowed_tickers: Sequence[str] | None = None,
    explicit_tickers: Sequence[str] | None = None,
):
    if explicit_tickers:
        tickers = tuple(str(ticker) for ticker in explicit_tickers)
        if not _protocol_universe_has_price_data(data_root, tickers, start=start, end=end):
            raise ValueError(
                f"--tickers {','.join(tickers)} has no price data in {start}..{end} under {data_root}"
            )
        return _UniverseSelection(
            tickers=tickers,
            label=f"explicit fixed universe: {len(tickers)} names",
        )
    protocol_universe = protocol.get("data_scope", {}).get("universe", [])
    if protocol_universe:
        tickers = tuple(str(ticker) for ticker in protocol_universe)
        if _protocol_universe_has_price_data(data_root, tickers, start=start, end=end):
            return _UniverseSelection(
                tickers=tickers,
                label=f"protocol fixed universe: {len(tickers)} names",
            )
    return select_fixed_universe(data_root, start, end, n=names, allowed_tickers=allowed_tickers)


def compare_market_state_labels_to_timeline_csv(
    market_states: Sequence[tuple[date, str]],
    timeline_csv: str | Path,
    *,
    start: date,
    end: date,
) -> dict[str, object]:
    expected: dict[str, str] = {}
    with Path(timeline_csv).open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            day = _parse_date(str(row["date"]))
            if start <= day <= end:
                expected[day.isoformat()] = str(row["state"])

    observed = {
        _to_iso_date(day): str(state)
        for day, state in market_states
        if start <= _parse_date(_to_iso_date(day)) <= end
    }
    dates = sorted(set(expected) | set(observed))
    mismatches = [
        {
            "date": day,
            "timeline_state": expected.get(day),
            "holdout_state": observed.get(day),
        }
        for day in dates
        if expected.get(day) != observed.get(day)
    ]
    result = {"matched": not mismatches, "n_checked": len(dates), "mismatches": mismatches}
    if mismatches:
        raise ValueError(f"market-state label mismatch against timeline CSV: {mismatches[:5]}")
    return result


def _to_iso_date(value) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _corpus_stats_checked(view_corpus, args) -> dict:
    """Corpus statistics for the manifest, refusing a replay that never hit the corpus.

    A missing view is a LEGAL abstention in this design, so a replay whose prompts no longer
    match silently degrades into a smaller committee instead of failing: every affected seat
    returns its data-unavailable default and never calls the backend. That happened on
    2026-07-28, when the extension names were replayed without their per-name news-sentiment
    cache and two of three seats abstained on every date while the run reported success.
    Zero hits in a replay mode is therefore an error, not a statistic.
    """
    stats = {
        "path": str(args.view_corpus),
        "mode": args.view_corpus_mode,
        "entries": len(view_corpus),
        "hits": view_corpus.hits,
        "records_written": view_corpus.records_written,
    }
    if args.view_corpus_mode in ("replay", "replay_by_date") and view_corpus.hits == 0:
        raise SystemExit(
            f"view corpus replay served ZERO views from {args.view_corpus} "
            f"({len(view_corpus)} entries loaded). Every language-model seat fell back to its "
            "data-unavailable default, so this run evaluated a smaller committee than the one "
            "recorded. Check that the run passes the same inputs as the recording -- most "
            "often a per-name --news-sentiment-cache."
        )
    return stats


def _store_from_snapshot(snapshot: dict, *, temperature: float | None = None) -> ExpertScorecardStore:
    """Rebuild a frozen scorecard for the sensitivity re-run.

    Must be ExpertScorecardStore.from_snapshot: the arm snapshots carry the
    scorecard moments (EW mean/var, hit rate, counts) plus the bounded
    reflection deltas. Rebuilding through the retired additive-delta
    ExpertWeightStore silently dropped the moments and made every
    sensitivity arm equal-weight.
    """
    return ExpertScorecardStore.from_snapshot(dict(snapshot), temperature=temperature)


def _frozen_cost_sensitivity_payload(
    *,
    result: ExpertWeightHoldoutResult,
    loader,
    cfg: BacktestConfig,
    experts,
    market_state_config: MarketStateConfig,
    signal_sources,
    sensitivity_cost_bps_per_side: float,
    reflect_every: int,
    reflection_window: int,
    reflection_min_support: int,
    attribution_mode: str = "raw",
    bounded_memory: bool = True,
    exit_threshold: float | None = None,
    market_proxy=None,
    parallel_expert_calls: bool = False,
    reflection=None,
    attribution_horizon: int = 1,
    confidence_weighted_vote: bool = False,
    soft_abstain_weight: float = 0.0,
    decision_cadence: int = 1,
    position_sizing: str = "binary",
    stale_view_abstain: int = 0,
) -> dict:
    test_cfg = replace(
        cfg,
        date_from=date.fromisoformat(str(result.split_protocol["test_start"])),
        date_to=date.fromisoformat(str(result.split_protocol["test_end"])),
        execution_cost_bps_per_side=float(sensitivity_cost_bps_per_side),
    )
    arms = {}
    for name, snapshot in result.arm_memory_snapshots.items():
        # The online arm's stored snapshot is its END state (learned through
        # the test window); replaying the window from it is lookahead. Its
        # sensitivity run reproduces the arm instead: seed at the pre-test
        # freeze and learn through the window again -- learning is
        # action-independent and cost-free, so the trajectory is identical
        # and only the P&L is re-priced.
        online = name == "market_state_memory_online"
        seed = dict(
            result.arm_memory_snapshots["market_state_memory"] if online else snapshot
        )
        if online:
            # The pre-test payload carries frozen=True; the reproduction run
            # must relearn through the window, so its seed thaws.
            seed.pop("frozen", None)
        store = _store_from_snapshot(
            seed, temperature=float(result.selected_temperature)
        )
        loop_result = run_expert_loop(
            loader,
            test_cfg,
            experts,
            market_state_config=market_state_config,
            weight_store=store,
            reflection=(reflection if online else None),
            reflect_every=reflect_every,
            reflection_window=reflection_window,
            reflection_min_support=reflection_min_support,
            decision_threshold=result.selected_decision_threshold,
            exit_threshold=exit_threshold,
            position_sizing=position_sizing,
            stale_view_abstain=stale_view_abstain,
            allow_reflection=online,
            signal_sources=signal_sources,
            attribution_mode=attribution_mode,
            attribution_horizon=attribution_horizon,
            confidence_weighted_vote=confidence_weighted_vote,
            soft_abstain_weight=soft_abstain_weight,
            decision_cadence=decision_cadence,
            market_proxy=market_proxy,
            parallel_expert_calls=parallel_expert_calls,
            run_id=f"expert_weight_sensitivity_{name}",
        )
        arms[name] = {
            "metrics": _metrics(loop_result, result.trial_count),
            "n_dates": loop_result.n_dates,
            "n_active_dates": loop_result.n_active_dates,
            "participation_rate": loop_result.participation_rate,
        }

    comparison_input = {
        name: type("Arm", (), {"metrics": payload["metrics"]})()
        for name, payload in arms.items()
    }
    return {
        "label": "sensitivity",
        "cost_bps_per_side": float(sensitivity_cost_bps_per_side),
        "arms": arms,
        "arm_comparison_vs_equal_weight": _arm_comparison_vs_equal_weight(comparison_input),
        "selected_decision_threshold": result.selected_decision_threshold,
    }


def _daily_series(loop_result) -> list[list]:
    """Per-date [iso_date, mean_gross_return, long_exposure] from attributed episodes.

    Gross = sum of the reporting-attribution components (market + selection when long, zero
    when flat). Multi-name runs average gross across tickers per date; exposure = fraction of
    names long. Costs are NOT applied here — the portfolio script charges them from exposure
    changes so every arm gets the identical cost model.
    """
    from trading_agent.memory.expert_reflection import ATTRIBUTION_COMPONENTS

    by_date: dict = {}
    for ep in loop_result.episodes:
        gross = sum(float(ep.attribution.get(c, 0.0)) for c in ATTRIBUTION_COMPONENTS)
        bucket = by_date.setdefault(ep.decision_date, [0.0, 0, 0])
        bucket[0] += gross
        bucket[1] += 1
        bucket[2] += 1 if ep.action == "long" else 0
    return [
        [d.isoformat(), total / n, longs / n]
        for d, (total, n, longs) in sorted(by_date.items())
    ]


def export_daily_views(result: ExpertWeightHoldoutResult, path: str, manifest: dict) -> str:
    """Write the equal arm's per-date expert views as JSONL (one line per date+ticker).

    Carries direction, confidence, safe_default and the expert's stated reasoning plus
    the causal market_state label — exactly what the committee-level decision layer saw,
    in text form. The FIRST line is an export-manifest header binding the file to its
    source run (corpus path, window, and whether prior lessons touched the prompts), so
    a consumer can refuse an export from the wrong configuration. Consumers: the
    LLM-decision-layer comparison arm. Export only; nothing in the decision path reads this.
    """
    records = result.arms["equal_weight"].loop_result.expert_view_records
    by_key: dict = {}
    for expert_id, rows in records.items():
        for row in rows:
            key = (row["date"], row["ticker"])
            entry = by_key.setdefault(key, {
                "date": row["date"].isoformat(),
                "ticker": row["ticker"],
                "market_state": row["market_state"],
                "views": {},
            })
            entry["views"][expert_id] = {
                "direction": row["direction"],
                "confidence": float(row["confidence"]),
                "safe_default": bool(row["safe_default"]),
                "reasoning": row.get("reasoning", ""),
            }
    header = {
        "__export_manifest__": {
            "view_corpus_path": (manifest.get("view_corpus") or {}).get("path"),
            "prior_lessons_path": manifest.get("prior_lessons_path"),
            "tickers": manifest.get("tickers"),
            "start": manifest.get("start"),
            "boundary": manifest.get("boundary"),
            "end": manifest.get("end"),
            "expert_model": manifest.get("expert_model"),
            "expert_model_revision": manifest.get("expert_model_revision"),
            "vote_policy": manifest.get("vote_policy"),
            "run_label": manifest.get("run_label"),
        }
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for key in sorted(by_key):
            fh.write(json.dumps(by_key[key], sort_keys=True) + "\n")
    return str(out)


def result_to_payload(
    result: ExpertWeightHoldoutResult,
    manifest: dict,
    *,
    sensitivity_payload: dict | None = None,
) -> dict:
    if int(manifest.get("configuration_trial_count", result.configuration_trial_count)) != int(
        result.configuration_trial_count
    ):
        raise ValueError("manifest configuration_trial_count must match result configuration_trial_count")
    if int(result.trial_count) < int(result.configuration_trial_count):
        # An ENLARGED budget is the pre-registered cumulative accounting
        # (--dsr-trial-count); anything below the run's own grid would
        # inflate every DSR and stays refused.
        raise ValueError("deflated Sharpe trial input must not undercut configuration_trial_count")
    # trial_count provenance is asserted structurally above; a numeric-equality
    # check against candidate_audit_count would false-positive whenever a run
    # happens to produce exactly len(threshold_grid) proposals.
    return {
        "mode": "holdout",
        "run_label": manifest.get("run_label", "HOLDOUT"),
        "system": SYSTEM_IDENTITY,
        "schema_version": REPORT_SCHEMA_VERSION,
        "manifest": manifest,
        "split_protocol": result.split_protocol,
        "freeze_ok": result.freeze_ok,
        "trial_count": result.trial_count,
        "configuration_trial_count": result.configuration_trial_count,
        "candidate_audit_count": result.candidate_audit_count,
        "selected_decision_threshold": result.selected_decision_threshold,
        "selected_temperature": result.selected_temperature,
        "validation_threshold_sweep": result.validation_threshold_sweep,
        "expert_error_by_state": result.expert_error_by_state,
        "weight_adjustments_by_state": result.weight_adjustments_by_state,
        "expert_adjustment_audit": list(result.expert_adjustment_audit),
        "expert_ablation_by_state": result.expert_ablation_by_state,
        "memory_snapshot": result.memory_snapshot,
        "arm_memory_snapshots": result.arm_memory_snapshots,
        "validation_arm_comparison_vs_equal_weight": result.validation_arm_comparison_vs_equal_weight,
        "arm_comparison_vs_equal_weight": result.arm_comparison_vs_equal_weight,
        "research_claims": result.research_claims,
        "system_completion": result.system_completion,
        "market_state_coverage": result.market_state_coverage,
        "market_state_labels": {
            "train": _market_state_labels(result.train_result),
            "validation": {
                name: _market_state_labels(arm.loop_result)
                for name, arm in result.validation_arms.items()
            },
            "arms": {
                name: _market_state_labels(arm.loop_result)
                for name, arm in result.arms.items()
            },
        },
        "per_state_arm_comparison_vs_equal_weight": _per_state_arm_comparison_vs_equal_weight(
            result,
            min_support=int(manifest["reflection_min_support"]),
        ),
        "underpowered_cells": _underpowered_cells(
            result,
            min_support=int(manifest["reflection_min_support"]),
        ),
        "expert_confidence_diagnostics": {
            "train": result.train_result.confidence_diagnostics,
            "validation": {
                name: arm.loop_result.confidence_diagnostics
                for name, arm in result.validation_arms.items()
            },
            "arms": {
                name: arm.loop_result.confidence_diagnostics
                for name, arm in result.arms.items()
            },
        },
        # E0 reliability diagnostic: expert calibration, directional behaviour and
        # cross-expert correlation by market state. Experts are regime-blind and
        # identical across arms, so the test-window view is taken from the
        # equal_weight arm to avoid double counting.
        "expert_reliability": {
            "train": expert_reliability_report(result.train_result.episodes),
            "test": expert_reliability_report(
                result.arms["equal_weight"].loop_result.episodes
            ),
        },
        # Design D3 neutral-usage instrumentation: direction mix per expert x state
        # from the RAW per-view records. Episodes exclude neutral abstentions and
        # safe defaults, so this key is the only place the neutral rate is visible.
        # Views are expert-level and identical across arms; the test window is taken
        # from the equal_weight arm (same convention as expert_reliability).
        "view_direction_summary": {
            "train": view_direction_summary(result.train_result.expert_view_records),
            "test": view_direction_summary(
                result.arms["equal_weight"].loop_result.expert_view_records
            ),
        },
        # E-SKILL: text lessons distilled from the training window at the freeze
        # point (audit + the fed arm's --prior-lessons-path input).
        "lessons_by_state": result.lessons_by_state,
        # BUY/HOLD/SELL action view (aligns reporting with mainstream single-name
        # agents); computed per arm from the long/flat targets.
        "action_summary": {
            name: buy_hold_sell_summary(arm.loop_result.episodes)
            for name, arm in result.arms.items()
        },
        # Per-date [iso_date, gross_return, exposure] per TEST arm — the input for the
        # portfolio-level primary metric (scripts/portfolio_arm_metrics.py). Gross is the
        # episode's reporting-attribution component sum (market + selection when long, 0 when
        # flat); costs are applied by the script from exposure changes, identically per arm.
        "daily_arm_series": {
            name: _daily_series(arm.loop_result)
            for name, arm in result.arms.items()
        },
        "per_name_episode_counts": {
            "train": _episode_counts_by_state_action(result.train_result),
            "validation": {
                name: _episode_counts_by_state_action(arm.loop_result)
                for name, arm in result.validation_arms.items()
            },
            "arms": {
                name: _episode_counts_by_state_action(arm.loop_result)
                for name, arm in result.arms.items()
            },
        },
        "episode_count_before_after": {
            "train": _episode_count_comparison(result.train_result),
            "validation": {
                name: _episode_count_comparison(arm.loop_result)
                for name, arm in result.validation_arms.items()
            },
            "arms": {
                name: _episode_count_comparison(arm.loop_result)
                for name, arm in result.arms.items()
            },
        },
        "per_state_arm_metrics": result.per_state_arm_metrics,
        "reflection_trial_audit": _jsonable(result.reflection_trial_audit),
        "verify_gate_report": _jsonable(result.verify_gate_report),
        "train": {
            "n_dates": result.train_result.n_dates,
            "n_active_dates": result.train_result.n_active_dates,
            "n_reflection_rounds": result.train_result.n_reflection_rounds,
            "n_proposals": result.train_result.n_proposals,
            "n_applied": result.train_result.n_applied,
            "n_learned": result.train_result.n_learned,
            "participation_rate": result.train_result.participation_rate,
        },
        "validation_arms": {
            name: {
                "metrics": dict(arm.metrics),
                "n_dates": arm.loop_result.n_dates,
                "n_active_dates": arm.loop_result.n_active_dates,
                "participation_rate": arm.loop_result.participation_rate,
            }
            for name, arm in result.validation_arms.items()
        },
        "arms": {
            name: {
                "metrics": dict(arm.metrics),
                "n_dates": arm.loop_result.n_dates,
                "n_active_dates": arm.loop_result.n_active_dates,
                "participation_rate": arm.loop_result.participation_rate,
            }
            for name, arm in result.arms.items()
        },
        "cost_sensitivity": sensitivity_payload,
    }


def payload_to_markdown(payload: dict) -> str:
    manifest = payload["manifest"]
    split = payload["split_protocol"]
    title = "Expert Weight Holdout"
    if manifest.get("run_label") and manifest["run_label"] != "HOLDOUT":
        title = f"{manifest['run_label']} {title}"
    lines = [
        f"# {title}",
        "",
        "## Manifest",
        f"- launcher: {manifest['launcher']}",
        f"- data_root: {manifest['data_root']}",
        f"- universe: {manifest['universe_label']}",
        f"- start: {manifest['start']}",
        f"- boundary: {manifest['boundary']}",
        f"- end: {manifest['end']}",
        f"- embargo_days: {manifest['embargo_days']}",
        f"- expert_backend: {manifest['expert_backend']}",
        f"- expert_model: {manifest['expert_model']}",
        f"- expert_model_revision: {manifest['expert_model_revision']}",
        f"- expert_transport: {manifest['expert_transport']}",
        f"- expert_temperature: {manifest['expert_temperature']}",
        f"- run_label: {manifest.get('run_label', 'HOLDOUT')}",
        f"- tickers: {', '.join(manifest['tickers'])}",
        f"- initial_cash: {manifest['execution']['initial_cash']}",
        f"- commission_per_share: {manifest['execution']['commission_per_share']}",
        f"- min_commission: {manifest['execution']['min_commission']}",
        f"- max_commission_rate: {manifest['execution']['max_commission_rate']}",
        f"- slippage_perc: {manifest['execution']['slippage_perc']}",
        f"- liquidity_cap_pct: {manifest['execution']['liquidity_cap_pct']}",
        f"- headline_cost_bps_per_side: {manifest['cost_protocol']['headline_bps_per_side']}",
        f"- sensitivity_cost_bps_per_side: {manifest['cost_protocol']['sensitivity_bps_per_side']}",
        f"- preload_date_from: {manifest['preload']['date_from']}",
        f"- freeze_ok: {payload['freeze_ok']}",
        f"- trial_count: {payload['trial_count']}",
        f"- configuration_trial_count: {payload['configuration_trial_count']}",
        f"- candidate_audit_count: {payload['candidate_audit_count']}",
        f"- selected_decision_threshold: {payload['selected_decision_threshold']}",
    ]
    if manifest.get("embargo_warning"):
        lines.append(
            "- WARNING: embargo_days below main-protocol minimum "
            f"({manifest['embargo_warning']['minimum_non_diagnostic_embargo_days']})"
        )
    lines.extend([
        "",
        "## Decision Threshold Selection",
        "| threshold | selected | primary_sharpe | primary_return | primary_turnover | primary_participation |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for threshold, row in payload["validation_threshold_sweep"].items():
        primary = row.get("arms", {}).get("market_state_memory", {}).get("metrics", {})
        lines.append(
            f"| {threshold} | {row.get('selected', False)} | "
            f"{primary.get('sharpe', 0.0):+.3f} | "
            f"{primary.get('total_return', 0.0):+.6f} | "
            f"{primary.get('turnover', 0.0):.4f} | "
            f"{primary.get('participation_rate', 0.0):.4f} |"
        )
    lines.extend([
        "",
        "## Expert Confidence Diagnostics",
        "| split | expert | n | real | real_rate | mean | min | max | long | flat | safe_default | data_unavailable | parse_error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for expert_id, diag in payload["expert_confidence_diagnostics"]["train"].items():
        reasons = diag.get("safe_default_reasons", {})
        lines.append(
            f"| train | {expert_id} | {diag['n']} | {diag.get('real_judgment_count', 0)} | "
            f"{diag.get('real_judgment_rate', 0.0):.4f} | {diag['mean']:.4f} | "
            f"{diag['min']:.4f} | {diag['max']:.4f} | "
            f"{diag['long_count']} | {diag['flat_count']} | {diag['safe_default_count']} | "
            f"{reasons.get('data_unavailable', 0)} | {reasons.get('parse_error', 0)} |"
        )
    lines.extend([
        "",
        "## Research Claims",
        f"- reflection_contribution: {payload['research_claims']['reflection_contribution']['comparison']}",
        (
            "- reflection_total_return_delta: "
            f"{payload['research_claims']['reflection_contribution'].get('total_return_delta', 0.0):+.6f}"
        ),
        f"- regime_conditioning_contribution: {payload['research_claims']['regime_conditioning_contribution']['comparison']}",
        (
            "- regime_total_return_delta: "
            f"{payload['research_claims']['regime_conditioning_contribution'].get('total_return_delta', 0.0):+.6f}"
        ),
        (
            "- best_expert_ablation: "
            f"{payload['research_claims']['best_expert_ablation']['expert_id']} "
            f"in {payload['research_claims']['best_expert_ablation']['market_state']} "
            f"signal_drop={payload['research_claims']['best_expert_ablation']['signal_component_drop']:+.6f}"
        ),
        f"- frozen_test_protocol: {payload['research_claims']['frozen_test_protocol']}",
        "",
        "## System Completion",
        f"- research_system_complete: {payload['system_completion']['research_system_complete']}",
        f"- scope: {payload['system_completion']['scope']}",
        (
            "- external_paper_validation_required: "
            f"{payload['system_completion']['external_paper_validation_required']}"
        ),
        f"- ready_for_defense_sections: {', '.join(payload['system_completion']['ready_for_defense_sections'])}",
        "",
        "| check | passed |",
        "|---|---:|",
    ])
    for name, passed in payload["system_completion"]["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend([
        "",
        "## Split Protocol",
        f"- train_start: {split['train_start']}",
        f"- train_end: {split['train_end']}",
        f"- validation_start: {split['validation_start'] or 'none'}",
        f"- validation_end: {split['validation_end'] or 'none'}",
        f"- test_start: {split['test_start']}",
        f"- test_end: {split['test_end']}",
        f"- embargo_days: {split['embargo_days']}",
        f"- memory_frozen_at: {split['memory_frozen_at']}",
        "",
        "## Train",
        f"- active dates: {payload['train']['n_active_dates']}",
        f"- reflection rounds: {payload['train']['n_reflection_rounds']}",
        f"- proposals: {payload['train']['n_proposals']}",
        f"- applied adjustments: {payload['train']['n_applied']}",
        f"- learned adjustments: {payload['train']['n_learned']}",
        "",
        "## Expert Error Summary",
        f"- train states: {', '.join(sorted(payload['expert_error_by_state'].get('train', {}))) or 'none'}",
        f"- adjustment states: {', '.join(sorted(payload['weight_adjustments_by_state'])) or 'none'}",
        "",
        "## Expert Adjustment Audit",
        "| state | expert | weight_delta | train_n | train_accuracy | train_error_score | signal_delta | participation_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["expert_adjustment_audit"]:
        lines.append(
            f"| {row['market_state']} | {row['expert_id']} | {row['weight_delta']:+.4f} | "
            f"{row['train_error_n']} | {row['train_accuracy']:.4f} | "
            f"{row['train_mean_error_score']:+.6f} | "
            f"{row['state_signal_component_delta']:+.6f} | "
            f"{row['state_participation_delta']:+.4f} |"
        )
    if payload["validation_arms"]:
        lines.extend([
            "",
            "## Validation Arms",
            "| arm | n_dates | n_active_dates | total_return | sharpe | max_drawdown | participation_rate | trial_count |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, arm in payload["validation_arms"].items():
            m = arm["metrics"]
            lines.append(
                f"| {name} | {arm['n_dates']} | {arm['n_active_dates']} | "
                f"{m['total_return']:+.6f} | {m['sharpe']:+.3f} | "
                f"{m['max_drawdown']:.6f} | {arm['participation_rate']:.4f} | "
                f"{m['trial_count']:.0f} |"
            )
        lines.extend([
            "",
            "## Validation Comparison vs Equal Weight",
            "| arm | total_return_delta | sharpe_delta | max_drawdown_delta | turnover_delta | participation_rate_delta | learned_adjustment_count_delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, comparison in payload["validation_arm_comparison_vs_equal_weight"].items():
            lines.append(
                f"| {name} | {comparison['total_return_delta']:+.6f} | "
                f"{comparison['sharpe_delta']:+.3f} | "
                f"{comparison['max_drawdown_delta']:+.6f} | "
                f"{comparison['turnover_delta']:+.4f} | "
                f"{comparison['participation_rate_delta']:+.4f} | "
                f"{comparison['learned_adjustment_count_delta']:+.0f} |"
            )
    lines.extend([
        "",
        "## Expert Ablation By State",
        "| removed_expert | state | full_signal | ablated_signal | signal_drop | full_participation | ablated_participation |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for expert_id, states in payload["expert_ablation_by_state"].items():
        for state, metrics in states.items():
            lines.append(
                f"| {expert_id} | {state} | "
                f"{metrics['full_mean_signal_component']:+.6f} | "
                f"{metrics['ablated_mean_signal_component']:+.6f} | "
                f"{metrics['signal_component_drop']:+.6f} | "
                f"{metrics['full_participation_rate']:.4f} | "
                f"{metrics['ablated_participation_rate']:.4f} |"
            )
    lines.extend([
        "",
        "## Holdout Arms",
        f"All figures in this section are net of headline transaction costs ({manifest['cost_protocol']['headline_bps_per_side']} bps per side).",
        "",
        "| arm | total_return | sharpe | deflated_sharpe | max_drawdown | turnover | participation_rate | trial_count | learned_adjustments |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, arm in payload["arms"].items():
        m = arm["metrics"]
        dsr = "n/a" if m["deflated_sharpe_ratio"] is None else f"{m['deflated_sharpe_ratio']:.4f}"
        lines.append(
            f"| {name} | {m['total_return']:+.6f} | {m['sharpe']:+.3f} | "
            f"{dsr} | {m['max_drawdown']:.6f} | {m['turnover']:.4f} | "
            f"{m['participation_rate']:.4f} | {m['trial_count']:.0f} | "
            f"{m['learned_adjustment_count']:.0f} |"
        )
    sensitivity = payload.get("cost_sensitivity")
    if sensitivity:
        lines.extend([
            "",
            "## Cost Sensitivity Arms",
            f"All figures in this section are net of sensitivity transaction costs ({sensitivity['cost_bps_per_side']} bps per side).",
            "",
            "| arm | total_return | sharpe | deflated_sharpe | max_drawdown | turnover | participation_rate | trial_count | learned_adjustments |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name, arm in sensitivity["arms"].items():
            m = arm["metrics"]
            dsr = "n/a" if m["deflated_sharpe_ratio"] is None else f"{m['deflated_sharpe_ratio']:.4f}"
            lines.append(
                f"| {name} | {m['total_return']:+.6f} | {m['sharpe']:+.3f} | "
                f"{dsr} | {m['max_drawdown']:.6f} | {m['turnover']:.4f} | "
                f"{m['participation_rate']:.4f} | {m['trial_count']:.0f} | "
                f"{m['learned_adjustment_count']:.0f} |"
            )
    lines.extend([
        "",
        "## Market State Coverage",
        "| split | arm | state | episode_count |",
        "|---|---|---|---:|",
    ])
    for state, count in payload["market_state_coverage"]["train"].items():
        lines.append(f"| train | train | {state} | {count} |")
    for arm_name, states in payload["market_state_coverage"]["arms"].items():
        for state, count in states.items():
            lines.append(f"| test | {arm_name} | {state} | {count} |")
    lines.extend([
        "",
        "## Per-Name Episode Counts",
        "| split | arm | state | long | flat |",
        "|---|---|---|---:|---:|",
    ])
    for state, actions in payload["per_name_episode_counts"]["train"].items():
        lines.append(f"| train | train | {state} | {actions.get('long', 0)} | {actions.get('flat', 0)} |")
    for arm_name, states in payload["per_name_episode_counts"]["arms"].items():
        for state, actions in states.items():
            lines.append(
                f"| test | {arm_name} | {state} | {actions.get('long', 0)} | {actions.get('flat', 0)} |"
            )
    lines.extend([
        "",
        "## Episode Count Before/After Refactor",
        "| split | arm | state | before_universe_level | after_per_name | increase |",
        "|---|---|---|---:|---:|---:|",
    ])
    for state, counts in payload["episode_count_before_after"]["train"].items():
        lines.append(
            f"| train | train | {state} | {counts['before_universe_level']} | "
            f"{counts['after_per_name']} | {counts['increase']} |"
        )
    for arm_name, states in payload["episode_count_before_after"]["arms"].items():
        for state, counts in states.items():
            lines.append(
                f"| test | {arm_name} | {state} | {counts['before_universe_level']} | "
                f"{counts['after_per_name']} | {counts['increase']} |"
            )
    lines.extend([
        "",
        "## Per-State Arm Metrics",
        "| arm | state | episode_count | mean_market_return | mean_signal_component | participation_rate |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for arm_name, states in payload["per_state_arm_metrics"].items():
        for state, metrics in states.items():
            lines.append(
                f"| {arm_name} | {state} | {metrics['episode_count']:.0f} | "
                f"{metrics['mean_market_return']:+.6f} | "
                f"{metrics['mean_signal_component']:+.6f} | "
                f"{metrics['participation_rate']:.4f} |"
            )
    lines.extend([
        "",
        "## Underpowered Cells",
        "| arm | state | episode_count | long | flat | min_support | absent | underpowered | excluded_from_headline |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["underpowered_cells"]:
        lines.append(
            f"| {row['arm']} | {row['state']} | {row['episode_count']} | "
            f"{row['long_count']} | {row['flat_count']} | {row['min_support']} | "
            f"{row.get('absent', False)} | {row['underpowered']} | "
            f"{row['exclude_from_headline_comparison']} |"
        )
    lines.extend([
        "",
        "## Per-State Comparison vs Equal Weight",
        "| arm | state | market_return_delta | signal_component_delta | participation_delta | underpowered | excluded_from_headline |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for arm_name, states in payload["per_state_arm_comparison_vs_equal_weight"].items():
        for state, metrics in states.items():
            lines.append(
                f"| {arm_name} | {state} | "
                f"{metrics['mean_market_return_delta']:+.6f} | "
                f"{metrics['mean_signal_component_delta']:+.6f} | "
                f"{metrics['participation_rate_delta']:+.4f} | "
                f"{metrics['underpowered']} | {metrics['exclude_from_headline_comparison']} |"
            )
    lines.extend([
        "",
        "## Comparison vs Equal Weight",
        "| arm | total_return_delta | sharpe_delta | max_drawdown_delta | turnover_delta | participation_rate_delta | learned_adjustment_count_delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, comparison in payload["arm_comparison_vs_equal_weight"].items():
        lines.append(
            f"| {name} | {comparison['total_return_delta']:+.6f} | "
            f"{comparison['sharpe_delta']:+.3f} | "
            f"{comparison['max_drawdown_delta']:+.6f} | "
            f"{comparison['turnover_delta']:+.4f} | "
            f"{comparison['participation_rate_delta']:+.4f} | "
            f"{comparison['learned_adjustment_count_delta']:+.0f} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(payload: dict, output_base: str | Path) -> tuple[Path, Path]:
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(payload_to_markdown(payload), encoding="utf-8")
    return json_path, md_path


def write_memory_snapshot(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload["memory_snapshot"], indent=2, sort_keys=True), encoding="utf-8")
    return path


def _require_date_window(args: argparse.Namespace) -> None:
    if args.validation_boundary is not None:
        if not (args.start < args.validation_boundary < args.boundary < args.end):
            raise ValueError(
                "--start must be before --validation-boundary, before --boundary, "
                "and --boundary before --end"
            )
        return
    if not (args.start < args.boundary < args.end):
        raise ValueError("--start must be before --boundary, and --boundary before --end")


def _require_positive_args(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        if getattr(args, name) <= 0:
            flag = f"--{name.replace('_', '-')}"
            raise ValueError(f"{flag} must be positive")


def _require_non_negative_args(args: argparse.Namespace, names: Sequence[str]) -> None:
    for name in names:
        if getattr(args, name) < 0:
            flag = f"--{name.replace('_', '-')}"
            raise ValueError(f"{flag} must be non-negative")


def _require_experiment_parameters(args: argparse.Namespace) -> None:
    _require_positive_args(
        args,
        (
            "names",
            "reflect_every",
            "reflection_window",
            "reflection_min_support",
            "initial_cash",
            "vol_window",
            "percentile_window",
        ),
    )
    _require_non_negative_args(
        args,
        (
            "embargo_days",
            "commission_per_share",
            "min_commission",
            "max_commission_rate",
            "slippage_perc",
            "headline_cost_bps_per_side",
            "sensitivity_cost_bps_per_side",
            "liquidity_cap_pct",
            "preload_calendar_days",
        ),
    )


def _expert_model_revision(args: argparse.Namespace) -> str:
    raw = getattr(args, "expert_model_revision", None)
    if raw:
        return str(raw)
    backend = str(args.expert_backend)
    if backend in {"openai", "codex_sdk"}:
        return str(args.expert_model)
    return f"{backend}-deterministic"


def _require_expert_model_identity(args: argparse.Namespace) -> None:
    if not getattr(args, "expert_model", None):
        args.expert_model = (
            DEFAULT_CODEX_SDK_MODEL
            if str(args.expert_backend).strip() == "codex_sdk"
            else DEFAULT_EXPERT_MODEL
        )
    args.expert_model_revision = _expert_model_revision(args)
    if not str(args.expert_backend).strip():
        raise ValueError("--expert-backend is required")
    if not str(args.expert_model).strip():
        raise ValueError("--expert-model is required")
    if not str(args.expert_model_revision).strip():
        raise ValueError("--expert-model-revision is required")


def _embargo_warning(args: argparse.Namespace) -> dict | None:
    minimum = 5
    if int(args.embargo_days) >= minimum:
        return None
    return {
        "level": "WARNING",
        "message": "embargo_days below main-protocol minimum for non-diagnostic runs",
        "embargo_days": int(args.embargo_days),
        "minimum_non_diagnostic_embargo_days": minimum,
    }


def main(argv: Sequence[str] | None = None) -> int:
    # Load secrets from a gitignored .env (OPENAI_API_KEY etc.) so keys live in a
    # file the user fills locally, never in source or on the command line. Uses
    # os.environ.setdefault, so an already-exported OPENAI_API_KEY still wins.
    try:
        from data_collection.config import load_env_file
        load_env_file()
    except Exception:
        pass
    args = build_parser().parse_args(argv)
    if args.temperature_grid is None:
        # tau lives on different scales for the two retrieval modes: a pure number
        # under standardization, the daily-return scale for raw q/tau.
        args.temperature_grid = (0.5, 1.0, 2.0) if args.standardized_memory else (0.0005, 0.001, 0.002)
    _require_date_window(args)
    _require_experiment_parameters(args)
    _require_expert_model_identity(args)
    args.preload_start = _preload_start(args.start, args.preload_calendar_days)
    data_root = str(resolve_data_root(args.data_root))
    protocol = _read_protocol(args.protocol)
    selection = _resolve_universe_selection(
        data_root,
        start=args.start,
        end=args.end,
        names=args.names,
        protocol=protocol,
        allowed_tickers=(
            available_tickers_from_sqlite_qlib_factors(args.alpha158_factors_db)
            if args.alpha158_factors_db
            else None
        ),
        explicit_tickers=args.tickers,
    )
    loader = load_panel(data_root, args.preload_start, args.end, tickers=selection.tickers, modalities=("price",))
    cfg = _config(args, selection.tickers)
    news_sentiment_backend = (
        OpenAIChatBackend(model=args.news_sentiment_model, temperature=0.0)
        if args.news_sentiment_cache
        else None
    )
    # Design D: technical = deterministic TradingView rating computed on the already-loaded
    # split-adjusted OHLCV panel (point-in-time by construction; no LLM call, no corpus).
    from trading_agent.alpha.technical_rating import TradingViewRating
    from trading_agent.market import LoaderOHLCVPanel

    technical_rating = TradingViewRating(LoaderOHLCVPanel(loader))
    signal_sources = build_expert_signal_sources(
        data_root=data_root,
        start=args.preload_start,
        end=args.end,
        tickers=selection.tickers,
        alpha158_factors_db=args.alpha158_factors_db,
        qlib_provider_uri=args.qlib_provider_uri,
        news_sentiment_backend=news_sentiment_backend,
        news_sentiment_cache_path=args.news_sentiment_cache,
        include_news=not args.no_news,
        strict_news=args.strict_news,
        include_filings=not args.no_filings,
        sec_facts_path=args.sec_facts_path,
        technical_rating=technical_rating,
        fred_macro_path=args.fred_macro_path,
        prior_lessons_path=args.prior_lessons_path,
    )
    if sum((args.vol_quartile_regime, args.trend_only_regime, args.risk_binary_regime,
            args.vol_tercile_regime, args.trend_cond_risk_regime, args.down_merged_regime)) > 1:
        raise SystemExit("--vol-quartile-regime, --trend-only-regime and --risk-binary-regime "
                         "are mutually exclusive")
    regime_mode = (
        "vol_tercile" if args.vol_tercile_regime
        else "trend_cond_risk" if args.trend_cond_risk_regime
        else "down_merged" if args.down_merged_regime
        else "vol_quartile" if args.vol_quartile_regime
        else "trend_only" if args.trend_only_regime
        else "risk_binary" if args.risk_binary_regime
        else "trend_risk"
    )
    market_state_config = MarketStateConfig(
        vol_window=args.vol_window,
        percentile_window=args.percentile_window,
        trend_window=args.trend_window,
        trend_confirm_days=args.trend_confirm_days,
        mode=regime_mode,
    )
    built_experts = build_experts(
        args.expert_backend,
        model=args.expert_model,
        model_revision=args.expert_model_revision,
        participation_prior=args.participation_prior,
    )
    view_corpus = None
    if args.view_corpus:
        view_corpus = ViewCorpus(args.view_corpus, mode=args.view_corpus_mode)
        attach_corpus(built_experts, view_corpus)
    if args.vote_policy == "binary":
        # Frozen binary replay over the SAME corpus: deliberate neutrals map to
        # the forced_direction captured in the same response. Wrap LAST so
        # corpus caching keys on the inner experts' raw prompts.
        from trading_agent.experts.backends import ForcedBinaryExpert

        built_experts = [ForcedBinaryExpert(e) for e in built_experts]
    market_proxy = None
    if args.market_proxy:
        from trading_agent.data.market_proxy import MarketProxy
        market_proxy = MarketProxy.from_parquet(args.market_proxy)
    elif len(selection.tickers) == 1:
        # HARD GUARD (2026-07-18): a single-name run's "equal-weight universe"
        # IS the traded stock, so the market_state label would be derived from
        # the stock itself — the exact leakage the market-state rule bans. The help text
        # said "required"; now it is enforced instead of silently degrading.
        raise SystemExit(
            "--market-proxy is REQUIRED for single-name runs: without it the "
            "market-state label falls back to the traded stock's own series "
            f"(universe = {selection.tickers}). Pass e.g. "
            "--market-proxy data/market_proxy/spy_daily.parquet"
        )
    reflection = None
    reflection_model = None
    if args.reflection_backend == "deterministic_scaled":
        from trading_agent.memory.expert_reflection import DeterministicExpertReflection

        reflection = DeterministicExpertReflection(
            min_support=args.reflection_min_support,
            step_mode=args.reflection_step_mode,
        )
    if args.reflection_backend == "deterministic":
        from trading_agent.memory.expert_reflection import DeterministicExpertReflection

        # Same maker as deterministic_scaled; the flag decides the step rule.
        # Constructed explicitly so the manifest's recorded step mode is the
        # one that actually ran (it used to fall back silently).
        reflection = DeterministicExpertReflection(
            min_support=args.reflection_min_support,
            step_mode=args.reflection_step_mode,
        )
    if args.reflection_backend == "llm":
        from trading_agent.memory.expert_reflection import LLMExpertReflection

        reflection_model = args.reflection_model or args.expert_model
        reflection = LLMExpertReflection(
            OpenAIChatBackend(model=reflection_model, temperature=0.0).call,
            min_support=args.reflection_min_support,
        )
    result = run_expert_weight_holdout(
        loader,
        cfg,
        built_experts,
        validation_boundary=args.validation_boundary,
        boundary=args.boundary,
        embargo_days=args.embargo_days,
        market_state_config=market_state_config,
        reflection=reflection,
        reflect_every=args.reflect_every,
        reflection_window=args.reflection_window,
        reflection_min_support=args.reflection_min_support,
        decision_threshold_grid=args.decision_thresholds,
        temperature_grid=args.temperature_grid,
        memory_half_life=args.memory_half_life,
        variance_penalty=args.variance_penalty,
        memory_score=args.memory_score,
        memory_shrinkage=args.memory_shrinkage,
        position_sizing=args.position_sizing,
        stale_view_abstain=args.stale_view_abstain,
        dsr_trial_count=args.dsr_trial_count,
        signal_sources=signal_sources,
        attribution_mode=args.attribution_mode,
        attribution_horizon=args.attribution_horizon,
        bounded_memory=not args.unbounded_memory,
        # Sealed configuration (2026-07-21): the bounded reflection delta stays LIVE in
        # verified cells — the canonical artefacts carry it at the selected temperature
        # (e.g. BIIB global +0.10/-0.05/-0.10, max weight shift 0.041). The prior
        # numeric-only default rested on an inertness claim that held only at the
        # saturated snapshot tau=0.001 and is retracted. --numeric-only-memory remains
        # the explicit delta-off ablation.
        use_reflection=not args.numeric_only_memory,
        certified_only_memory=args.certified_only_memory,
        standardize_memory=args.standardized_memory,
        shrink_to_global=args.shrink_to_global,
        exit_threshold=args.exit_threshold,
        market_proxy=market_proxy,
        parallel_expert_calls=args.parallel_experts,
        verify_gate=args.terminal_verify,
        verify_alpha=args.verify_alpha,
        verify_correction=args.verify_correction,
        verify_persistence=args.verify_persistence,
        verify_min_effect=args.verify_min_effect,
        verify_oos=args.verify_oos,
        verify_kfold=args.verify_kfold,
        verify_block=args.verify_block,
        confidence_weighted_vote=args.confidence_weighted_vote,
        soft_abstain_weight=args.soft_abstain_weight,
        decision_cadence=args.decision_cadence,
    )
    sensitivity_payload = None
    if args.sensitivity_cost_bps_per_side != args.headline_cost_bps_per_side:
        sensitivity_payload = _frozen_cost_sensitivity_payload(
            result=result,
            loader=loader,
            cfg=cfg,
            experts=built_experts,
            market_state_config=market_state_config,
            signal_sources=signal_sources,
            sensitivity_cost_bps_per_side=args.sensitivity_cost_bps_per_side,
            exit_threshold=args.exit_threshold,
            reflect_every=args.reflect_every,
            reflection_window=args.reflection_window,
            reflection_min_support=args.reflection_min_support,
            attribution_mode=args.attribution_mode,
            bounded_memory=not args.unbounded_memory,
            market_proxy=market_proxy,
            reflection=reflection,
            attribution_horizon=args.attribution_horizon,
            confidence_weighted_vote=args.confidence_weighted_vote,
            soft_abstain_weight=args.soft_abstain_weight,
            decision_cadence=args.decision_cadence,
            position_sizing=args.position_sizing,
            stale_view_abstain=args.stale_view_abstain,
            parallel_expert_calls=args.parallel_experts,
        )
    manifest = {
        "mode": "holdout",
        "system": SYSTEM_IDENTITY,
        "schema_version": REPORT_SCHEMA_VERSION,
        "data_root": data_root,
        "universe_label": selection.label,
        "tickers": list(selection.tickers),
        "start": args.start.isoformat(),
        "validation_boundary": args.validation_boundary.isoformat() if args.validation_boundary else None,
        "boundary": args.boundary.isoformat(),
        "end": args.end.isoformat(),
        "embargo_days": args.embargo_days,
        "run_label": str(args.run_label).upper(),
        "protocol": args.protocol,
        "reserved_test_periods": protocol.get("reserved_test_periods", []),
        "seed": args.seed,
        "reflect_every": args.reflect_every,
        "reflection_window": args.reflection_window,
        "reflection_min_support": args.reflection_min_support,
        "reflection_backend": args.reflection_backend,
        "reflection_step_mode": args.reflection_step_mode,
        "memory_half_life": args.memory_half_life,
        "dsr_trial_count": args.dsr_trial_count,
        "variance_penalty_override": args.variance_penalty,
        "memory_score": args.memory_score,
        "memory_shrinkage": args.memory_shrinkage,
        "position_sizing": args.position_sizing,
        "stale_view_abstain": args.stale_view_abstain,
        "terminal_verify": bool(args.terminal_verify),
        "verify_alpha": float(args.verify_alpha),
        "verify_correction": args.verify_correction,
        "verify_persistence": bool(args.verify_persistence),
        "verify_min_effect": args.verify_min_effect,
        "verify_oos": bool(args.verify_oos),
        "verify_kfold": int(args.verify_kfold),
        "verify_block": args.verify_block,
        "reflection_model": reflection_model,
        "reflection_fallback_count": (
            int(getattr(reflection, "fallback_count", 0)) if reflection is not None else None
        ),
        "use_reflection": not args.numeric_only_memory,
        "reflection_under_certified": bool(args.reflection_under_certified),
        "certified_only_memory": args.certified_only_memory,
        "participation_prior": args.participation_prior,
        "standardized_memory": args.standardized_memory,
        "shrink_to_global": args.shrink_to_global,
        "attribution_mode": args.attribution_mode,
        "attribution_horizon": args.attribution_horizon,
        "bounded_memory": not args.unbounded_memory,
        "exit_threshold": args.exit_threshold,
        "vote_policy": args.vote_policy,
        "confidence_weighted_vote": bool(args.confidence_weighted_vote),
        "soft_abstain_weight": float(args.soft_abstain_weight),
        "decision_cadence": int(args.decision_cadence),
        "prior_lessons_path": args.prior_lessons_path,
        "parallel_expert_calls": bool(args.parallel_experts),
        "market_proxy": args.market_proxy,
        "regime_source": "spy_proxy" if args.market_proxy else "equal_weight_universe",
        "regime_mode": regime_mode,
        "decision_thresholds": [float(x) for x in args.decision_thresholds],
        "temperature_grid": [float(x) for x in args.temperature_grid],
        "preload": {
            "calendar_days": int(args.preload_calendar_days),
            "date_from": args.preload_start.isoformat(),
        },
        "market_state": {
            "vol_window": args.vol_window,
            "percentile_window": args.percentile_window,
            "trend_window": args.trend_window,
            "trend_confirm_days": args.trend_confirm_days,
        },
        "execution": {
            # Values from the effective BacktestConfig: when the bps model is
            # active the commission parameters are zeroed because they are not
            # charged (bps replaces per-share commission in the adapter).
            "initial_cash": cfg.initial_cash,
            "commission_per_share": cfg.commission_per_share,
            "min_commission": cfg.min_commission,
            "max_commission_rate": cfg.max_commission_rate,
            "slippage_perc": cfg.slippage_perc,
            "execution_cost_bps_per_side": cfg.execution_cost_bps_per_side,
            "liquidity_cap_pct": cfg.liquidity_cap_pct,
        },
        "cost_protocol": {
            "model": (
                "bps_per_side_on_traded_notional"
                if cfg.execution_cost_bps_per_side > 0
                else "per_share_commission"
            ),
            # True only while a positive bps rate is charged, which is what displaces
            # the per-share schedule. It was hardcoded, so a run under the commission
            # model reported that its commission had been replaced by a rate of zero.
            "bps_replaces_per_share_commission": cfg.execution_cost_bps_per_side > 0,
            "headline_bps_per_side": float(args.headline_cost_bps_per_side),
            "sensitivity_bps_per_side": float(args.sensitivity_cost_bps_per_side),
            "arm_identical": True,
        },
        "expert_backend": args.expert_backend,
        "expert_model": args.expert_model,
        "expert_model_revision": args.expert_model_revision,
        **expert_backend_metadata(
            args.expert_backend,
            model=args.expert_model,
            model_revision=args.expert_model_revision,
        ),
        "view_corpus": (
            _corpus_stats_checked(view_corpus, args)
            if view_corpus is not None
            else None
        ),
        "configuration_trial_count": int(result.configuration_trial_count),
        "candidate_audit_count": int(result.candidate_audit_count),
        "embargo_warning": _embargo_warning(args),
        "launcher": args.launcher,
        "alpha158_factors_db": args.alpha158_factors_db,
        "qlib_provider_uri": args.qlib_provider_uri,
        "news_enabled": not args.no_news,
        "filings_enabled": not args.no_filings,
        "strict_news": bool(args.strict_news),
        "news_sentiment_cache": args.news_sentiment_cache,
        "news_sentiment_model": args.news_sentiment_model if args.news_sentiment_cache else None,
    }
    payload = result_to_payload(result, manifest, sensitivity_payload=sensitivity_payload)
    json_path, md_path = write_reports(payload, args.output)
    if args.memory_output:
        memory_path = write_memory_snapshot(payload, args.memory_output)
        print(f"WROTE {json_path}, {md_path}, and {memory_path}", flush=True)
    else:
        print(f"WROTE {json_path} and {md_path}", flush=True)
    if args.export_daily_views:
        export_path = export_daily_views(result, args.export_daily_views, manifest)
        print(f"WROTE {export_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
