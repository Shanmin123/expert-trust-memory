# Code map

Orientation for `src/trading_agent/`, file by file.

```text
src/trading_agent/
  cli.py                            # three-mode launcher: backtest, holdout, paper
  runner.py                         # the daily decision loop (run_expert_loop) and run_backtest
  expert_context.py                 # builds each expert's context from the data adapters
  contracts.py                      # Signal, ExpertView, the Expert protocol, dataclasses
  baselines.py                      # deterministic equal-weight engines for the smoke test
  reporting.py, persistence.py, market.py
  paper_trading.py                  # IB paper order preview and safety helpers
  paper_live_context.py             # adapts IB frames into the backtest's context builder
  paths.py                          # where the data lives, resolved from TRADING_AGENT_DATA_ROOT
  alpha/
    market_state_classifier.py      # the market-state label: SPY trend crossed with a volatility rank
    momentum.py, reversal.py, lowvol.py, technical_indicators.py
    alpha158_strategy.py, alpha158_recipes.py, alpha158_pit.py
    text_attention.py, llm_news_sentiment.py, filing_signals.py, fred_macro.py
    sec_fundamentals.py             # point-in-time SEC XBRL annual and quarterly ratios
  experts/
    base_expert.py                  # ExpertBackend protocol, prompt, parse, safe default
    technical_expert.py, news_expert.py, fundamental_expert.py, factor_expert.py
    backends.py                     # static, heuristic, openai and codex_sdk backends; build_experts
    view_corpus.py                  # recorded view corpus: record and replay (CachedExpertBackend)
  aggregation/
    weights.py                      # floored-simplex projection
    soft_weighting.py               # the weighted vote
  memory/
    expert_scorecard.py             # the expert-trust memory: scorecard records and softmax weights
    expert_reflection.py            # reflection into bounded adjustments, and the verification test
    expert_error.py                 # per-expert attributed error records
    expert_weights.py               # retired delta store; kept for MARKET_STATES and ExpertWeightEntry
    metrics.py                      # deflated Sharpe ratio
  execution/
    finsaber_adapter.py             # adapter onto the FINSABER execution formulas
  data/
    finsaber_panel.py, finsaber_news.py, finsaber_text.py, finsaber_filings.py, market_proxy.py
  experiments/
    expert_weight_holdout.py        # three-setting train, freeze and test evaluation
    expert_weight_holdout_launcher.py
    llm_decision_layer.py           # the text-form trader
    real_data_diagnostic.py         # signal-only diagnostic
    expert_reliability_diagnostic.py
src/data_collection/                # SEC XBRL, FRED and Interactive Brokers collectors
```

`factor_expert.py` and `alpha158_*` are in the tree but not in the committee: `build_experts` builds the
technical, news and fundamental experts only. The Alpha158 code feeds the factor-model baselines.

The scripts that drive experiments, baselines and figures are indexed in
[`../scripts/README.md`](../scripts/README.md).

## Paper-mode flags

`paper` builds an observation from the latest IB bars, labels the market state from the latest IB
daily bars by default, calls the expert views, loads frozen memory from `--memory-snapshot`, applies
the weights through aggregation, and scales the deployed cash by the aggregation score.

- `--order SYMBOL:buy|sell:QTY` or `--auto-decide --paper-cash <amount>`: where orders come from.
- `--paper-market-state <state>`: a manual override of the market state. It is recorded in the manifest.
- `--committee-read <dir or file>`: take the views from a recorded committee read instead of the static
  experts. See [`live_trading.md`](live_trading.md) for why this matters.
- `--require-memory`: forbids the equal-weight fallback on deployment runs.
- `--use-account-cash`: reads IB `AvailableFunds` instead of a supplied cash figure.
- `--max-order-value`: caps each generated preview by notional value.
- `--dry-run`: the manual form does not connect at all. With `--auto-decide` it connects for market
  data and account cash only, then disconnects.
- `--execute-orders`: the only path that submits to the connected IB paper account. It is never valid
  together with `--dry-run`.

## Per-expert signal files

- `technical`: `alpha/{momentum,reversal,lowvol,technical_indicators}.py`.
- `news`: `alpha/{text_attention,llm_news_sentiment}.py` and `data/finsaber_news.py`.
- `fundamental`: `alpha/sec_fundamentals.py` and the filing-text signals in `alpha/filing_signals.py`.

## Opt-in research variants

The holdout launcher carries several variants that are off by default, and the reported results are
produced with all of them off:

- Verification: `--verify-{alpha,correction,persistence,min-effect,oos,kfold}`.
- Decision layer: `--confidence-weighted-vote`, `--soft-abstain-weight`, `--decision-cadence`,
  `--position-sizing proportional`, `--stale-view-abstain`.
- Memory: `--memory-half-life`, `--variance-penalty`, `--memory-score hit_logodds`,
  `--reflection-step-mode t_stat`, `--raw-memory`, `--shrink-to-global`, `--numeric-only-memory`.
- Accounting: `--dsr-trial-count`, which can only enlarge the trial budget.

## Reporting set

Per stock and as a cross-stock average: cumulative and annualised return, annualised volatility,
maximum drawdown, Sharpe, Sortino and deflated Sharpe ratios, turnover, participation, and the per-state
attribution of each expert.
