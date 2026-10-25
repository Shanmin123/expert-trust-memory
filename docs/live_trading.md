# Running against an Interactive Brokers paper account

How the forward run was executed, and how to repeat a session. Orders go to a paper account only.

## What runs

Two stages, kept apart so that the analysis can never place an order.

1. `scripts/live_analysis.py` reads live bars and headlines, labels the market state from live SPY,
   runs the three experts and the frozen memory, and prints the committee's view for each stock. It
   imports no execution path. It also writes the day's committee read, which is the forward run's
   record, to `reports/paper_forward/reads/`.
2. `scripts/paper_forward_run.py` replays that read through `--committee-read`, one process per
   stock, compares the decision with the positions the account holds, and submits only the
   difference. A stock already at its target produces no order.

`scripts/forward_day.py` runs both stages in order for one trading day.

## Prerequisites

1. **IB Gateway running and logged in** to a paper account, with the API enabled, on port 4002
   (TWS paper trading uses 7497). The API is local, so the Gateway must run on the same host.
2. **One market-data session.** IB allows one. If TWS, IBKR Mobile or the Client Portal is logged in
   elsewhere, historical requests fail with error 162 and return an empty frame instead of an error.
3. `OPENAI_API_KEY` in `.env`, for the two model-backed experts. `--no-llm` skips them, and they
   abstain.
4. The `paper` extra: `pip install -e ".[paper]"`.

## Running a session

```bash
python scripts/live_analysis.py                       # the committee read for all twenty stocks
python scripts/live_analysis.py --symbols BIIB,ECL    # a subset
python scripts/live_analysis.py --no-llm              # technical expert only, no API spend

python scripts/paper_forward_run.py                   # print the twenty order commands
python scripts/paper_forward_run.py --preview         # run them: no order is submitted
python scripts/paper_forward_run.py --preview --execute   # submit the position changes
```

Submission needs both `--preview` and `--execute`, so the pass that sends an order has already
printed it. Each stock runs as its own process with its own IB client id, because a memory snapshot
carries no ticker. Manifests are filed under `reports/paper_forward/<decision date>/`, one directory
per session.

Two commands work without a Gateway:

```bash
python scripts/show_deployed_weights.py                                  # the weights each snapshot deploys
python scripts/live_analysis.py --offline reports/paper_forward/bars     # re-read bars that an earlier live run saved
```

## Safety

- Orders are previewed unless `--execute-orders` is passed to `python -m trading_agent paper`, and
  `--dry-run` and `--execute-orders` cannot be combined.
- Execution is restricted to the paper ports 4002 and 7497 unless `--allow-non-paper-port` is given.
- `--max-order-value` caps each order by notional. The forward run used $5,000, so the twenty
  positions together could not exceed $100,000.
- `--require-memory` fails instead of falling back to equal weights.
- The IB side, `ibkr-live-trading`, refuses to submit unless the account matches the configured
  paper-account prefix, `DU` by default.

## Three ways a live run goes quietly wrong

**1. Static experts.** Without `--committee-read`, `python -m trading_agent paper --auto-decide`
builds its views from `build_static_experts()`, whose backend always answers long. On 2026-08-03 that
produced buy previews on all twenty stocks, including four that the real committee had called
bearish. The order path must replay the committee read, which is what `paper_forward_run.py` does.
Each manifest names its views under `decision_source`: `committee_read:<file>` is the real committee,
and `three_expert_static` is the stub. A committee read dated for a different day than the run is
refused.

**2. History duration.** IB rejects a day-unit history request much beyond a year: asking for
`"792 D"` times out and returns an empty frame with no error. Measured on the paper Gateway, `1 Y`
returns 251 rows, `2 Y` 501, `3 Y` 751 and `792 D` none. Anything beyond a year is requested in years
(`_paper_history_duration_arg` in `cli.py`).

**3. Empty frames.** An empty market-proxy frame would make the classifier fall back to the traded
stock's own price history, so the label that describes the market would be computed from the
instrument being traded. On a deployment run (`--require-memory`) an empty frame raises. Keep that
behaviour in anything adapted from this code: a self-referential market state produces plausible
numbers and invalidates them.

## Where the pieces are

| File | Role |
|---|---|
| `scripts/live_analysis.py` | the whole analysis loop, from IB to the committee read |
| `scripts/paper_forward_run.py` | order previews and submission from a recorded read |
| `src/trading_agent/paper_live_context.py` | adapts IB frames into the backtest's own context builder, so a live decision runs through the same context pipeline: technical rating, point-in-time SEC fundamentals, macro block, news adapter |
| `src/trading_agent/cli.py`, `paper` mode | the order-preview path |
| `scripts/preview_decision.py` | one recorded day's decision, read back to its inputs; never trades |
