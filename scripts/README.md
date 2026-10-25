# Scripts

Experiment drivers, baselines and figure builders. Run them from the repository root with the
package installed (or with `PYTHONPATH=src`).

Scripts that need data read it from `TRADING_AGENT_DATA_ROOT` (see the README, "Data"). Scripts
later in the pipeline read the outputs of earlier ones from `reports/`, which is not tracked.
Steps that score a stock replay the recorded expert views, which are not released, so they stop
on the first missing input. Only `show_deployed_weights.py` needs nothing beyond what the
repository ships.

## Entry points

| Script | What it does |
|---|---|
| [`trading_agent_cli.py`](trading_agent_cli.py) | Launcher shim: the three modes without installing |
| [`run_expert_weight_holdout.py`](run_expert_weight_holdout.py) | Holdout entry point that the replay drivers call |

## Data collection

| Script | What it does |
|---|---|
| [`collect_sec_xbrl.py`](collect_sec_xbrl.py) | Point-in-time SEC XBRL financials |
| [`collect_fred_macro.py`](collect_fred_macro.py) | FRED macro block for the news expert |
| [`download_market_proxy.py`](download_market_proxy.py) | SPY proxy for the market-state label |
| [`select_random_universe.py`](select_random_universe.py) | The reproducible twenty-stock draw |

## Main result and statistics

| Script | What it does |
|---|---|
| [`portfolio20.py`](portfolio20.py) | Assemble and score the twenty-stock portfolio |
| [`portfolio_arm_metrics.py`](portfolio_arm_metrics.py) | Portfolio-level metrics, costs charged from exposure changes |
| [`per_stock_metrics.py`](per_stock_metrics.py) | The four reported metrics per stock |
| [`consolidate_cross_name.py`](consolidate_cross_name.py) | Per-stock reports into the cross-name tables |
| [`cross_name_significance.py`](cross_name_significance.py) | Cross-name significance tests |
| [`portfolio_block_sensitivity.py`](portfolio_block_sensitivity.py) | Bootstrap p-value against block length |
| [`rescore_finsaber_cost.py`](rescore_finsaber_cost.py) | Re-score under the FINSABER commission schedule |
| [`reprice_next_open.py`](reprice_next_open.py) | Reprice decisions at next-open execution |
| [`tau_sensitivity_audit.py`](tau_sensitivity_audit.py) | What the tuned temperature changes |
| [`verified_records_table.py`](verified_records_table.py) | Appendix table of every verified record |
| [`summarise_holdout_runs.py`](summarise_holdout_runs.py) | Cross-name summary over holdout reports |

## External baselines

| Script | What it does |
|---|---|
| [`baseline_arms.py`](baseline_arms.py) | Rule-based timing baselines |
| [`baseline_table.py`](baseline_table.py) | Score the baselines like the memory |
| [`baselines_table_full.py`](baselines_table_full.py) | The baseline table with the full metric set |
| [`buy_and_hold_nextopen.py`](buy_and_hold_nextopen.py) | Buy and hold on the same accounting |
| [`alpha_mining_baseline.py`](alpha_mining_baseline.py) | Alpha158 ridge and boosted-tree baselines |
| [`finmem_baseline.py`](finmem_baseline.py) | FinMem as a baseline |
| [`finmem_reprice_commission.py`](finmem_reprice_commission.py) | FinMem under the FINSABER commission schedule |
| [`finmem_implied_exposure.py`](finmem_implied_exposure.py) | FinMem's daily exposure from its equity curve |
| [`tradingagents_baseline.py`](tradingagents_baseline.py) | TradingAgents as a baseline |
| [`tradingagents_to_series.py`](tradingagents_to_series.py) | TradingAgents decisions into daily series |
| [`agent_baseline_table.py`](agent_baseline_table.py) | Per-stock comparison with the published agents |

## Ablations and sources of the effect

| Script | What it does |
|---|---|
| [`ungated_replay.py`](ungated_replay.py) | Replay a stock from its stored memory without verification |
| [`ungated_twenty.py`](ungated_twenty.py) | The learner without verification on the twenty stocks |
| [`ungated_verdict.py`](ungated_verdict.py) | What verification is worth |
| [`horizon_sweep20.py`](horizon_sweep20.py) | Attribution horizon against surviving records |
| [`trust_persistence.py`](trust_persistence.py) | Does measured trust persist across periods |
| [`run_market_state_timeline.py`](run_market_state_timeline.py) | The market-state timeline figure |

## Findings in text form

| Script | What it does |
|---|---|
| [`produce_llm_skills.py`](produce_llm_skills.py) | Model-written findings from verified records |
| [`producer_recovery.py`](producer_recovery.py) | Findings scored against the verified records |
| [`run_llm_decision_layer.py`](run_llm_decision_layer.py) | One text-trader comparison |
| [`run_trader_arms.py`](run_trader_arms.py) | The text-trader settings on the twenty stocks |
| [`score_primaries20.py`](score_primaries20.py) | Score the text-trader primaries |
| [`llm_decision_paired_analysis.py`](llm_decision_paired_analysis.py) | Paired analysis for the text trader |

## Forward run

| Script | What it does |
|---|---|
| [`live_analysis.py`](live_analysis.py) | Live committee read on the twenty stocks |
| [`paper_forward_run.py`](paper_forward_run.py) | IB paper forward-run commands |
| [`forward_day.py`](forward_day.py) | One forward-run trading day |
| [`forward_buy_and_hold.py`](forward_buy_and_hold.py) | Buy and hold over the forward window |
| [`forward_run_accounting.py`](forward_run_accounting.py) | The forward run's return by position |
| [`show_deployed_weights.py`](show_deployed_weights.py) | Weights each frozen snapshot deploys |
| [`preview_decision.py`](preview_decision.py) | One day's committee decision, never trades |

## Figures

| Script | What it does |
|---|---|
| [`architecture_figures.py`](architecture_figures.py) | System, protocol and data-pipeline diagrams |
| [`dissertation_figures.py`](dissertation_figures.py) | The result figures |
| [`equity_curve_figure.py`](equity_curve_figure.py) | The portfolio equity curve |
