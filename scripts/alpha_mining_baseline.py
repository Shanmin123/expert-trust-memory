"""An alpha-mining baseline: a model learned over the Alpha158 factor library.

Requested as a comparison object, not as a component of the system. Alpha158 is Qlib's
mined library of 158 formulaic price and volume alphas \\citep{qlib2021}; the mining step is
theirs, and what is learned here is the combination. Two models, both standard: ridge for
the linear case and histogram gradient boosting for the nonlinear one, which is the
substitute for Qlib's LightGBM recipe because lightgbm cannot import in this environment
(its compat layer pulls a dask that the installed pandas breaks).

Naming this honestly matters. It is a learned factor model over a mined alpha library, and
that is what the write-up should say. It is not formula search: no new alpha expressions
are discovered here.

Point-in-time discipline, since a factor model is where leakage usually enters:
  * features on date t are Alpha158 values, which are rolling functions of data up to t;
  * the label is the return from t to t+1, strictly forward;
  * the position decided on t earns the return of t+1, so the signal never reads its own
    outcome;
  * the model, and the feature scaler, are fit on training dates only. Fitting a scaler
    on the full sample is the classic silent leak and is guarded by a test.

Output is the arms' series shape with exposure and no cost applied, declared ``gross``, so
``baseline_table.py`` charges this project's ten basis points on the exposure path exactly
as it does for the committee.

    PYTHONPATH=src python scripts/alpha_mining_baseline.py
    PYTHONPATH=src python scripts/alpha_mining_baseline.py --model ridge --only DHI
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from trading_agent.paths import data_root, finsaber_prices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "baselines"


def factors_db() -> Path:
    """The Alpha158 factor table, in data/ or under TRADING_AGENT_DATA_ROOT.

    Found when a run needs it rather than at import, so the unit tests can load this
    module on a machine that has no data.
    """
    for base in (ROOT / "data", data_root()):
        path = base / "sp500_qlib_direct_factors.db"
        if path.exists():
            return path
    raise FileNotFoundError(
        "Alpha158 factors db not found in data/ or TRADING_AGENT_DATA_ROOT")


def panel() -> Path:
    return Path(str(finsaber_prices()))

TICKERS = ("TSLA", "ADSK", "BIIB", "CVX", "DHI", "ECL", "PNC", "TGT")
TRAIN_END = "2023-12-31"        # the memory freeze boundary
TEST_START = "2024-07-09"       # the sealed scored window
TEST_END = "2025-12-01"
SEED = 20260716


def factor_frame(ticker: str) -> pd.DataFrame:
    with sqlite3.connect(factors_db()) as con:
        df = pd.read_sql(f"select * from qlib_direct_{ticker}", con)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").set_index("date").drop(columns=["symbol"])


def price_returns(ticker: str) -> pd.Series:
    import pyarrow.dataset as ds
    table = ds.dataset(str(panel()), format="parquet", partitioning="hive").to_table(
        filter=(ds.field("symbol") == ticker))
    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").set_index("date")
    col = next(c for c in ("adjusted_close", "adj_close", "close") if c in df.columns)
    return df[col].pct_change().shift(-1).rename("fwd")   # return from t to t+1


def build_model(kind: str):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    if kind == "ridge":
        return Ridge(alpha=1.0, random_state=None)
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=200, learning_rate=0.05, random_state=SEED)


LLM_MODEL = "gpt-4.1-mini-2025-04-14"
LLM_CORPUS = OUT / "alpha_llm_selection_corpus.jsonl"


def llm_select_weights(ticker: str, train: pd.DataFrame, cols: list) -> tuple[dict, str]:
    """The large-model variant of the combination step: the model reads each factor's
    training-window rank IC (overall and by year) and returns a small weighted subset.

    The selection sees training statistics only, and the test window begins after the
    model's knowledge cutoff, the same anti-memorisation argument the experts carry.
    Responses are recorded once and replayed, keyed by a hash of the prompt.
    """
    import hashlib

    years = sorted({d[:4] for d in train.index})
    lines = []
    fwd = train["fwd"]
    for c in cols:
        s = train[c]
        overall = s.rank().corr(fwd.rank())
        by_year = []
        for y in years:
            m = train.index.str.startswith(y)
            if m.sum() >= 60:
                by_year.append(s[m].rank().corr(fwd[m].rank()))
        pos = sum(1 for v in by_year if v == v and v > 0)
        n = sum(1 for v in by_year if v == v)
        lines.append(f"{c.replace('QLIB158_', '')}: IC {overall:+.4f}, "
                     f"positive in {pos}/{n} years")

    prompt = (
        f"You select a small alpha combination from Qlib's Alpha158 library for one US "
        f"stock ({ticker}). Below is each factor's Spearman rank IC against the "
        f"next-day return over the training window {train.index[0]}..{train.index[-1]}, "
        f"with its sign stability across years. Choose AT MOST 8 factors you expect to "
        f"generalise out of sample, preferring stable signs over large one-sample ICs, "
        f"and assign each a weight; a negative weight trades against the factor. Return "
        f"ONLY JSON: {{\"weights\": {{\"FACTOR_NAME\": weight, ...}}}}.\n\n"
        + "\n".join(lines))
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if LLM_CORPUS.exists():
        for line in LLM_CORPUS.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") == key:
                return rec["weights"], "replayed"

    import os
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        env = ROOT / ".env"
        for ln in env.read_text(encoding="utf-8").splitlines():
            if ln.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = ln.split("=", 1)[1].strip()
    client = OpenAI()
    resp = client.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        messages=[{"role": "user", "content": prompt}])
    text = resp.choices[0].message.content.strip()
    text = text[text.index("{"):text.rindex("}") + 1]
    weights = {("QLIB158_" + k if not k.startswith("QLIB158_") else k): float(v)
               for k, v in json.loads(text)["weights"].items()}
    weights = {k: v for k, v in list(weights.items())[:8] if k in set(cols)}
    if not weights:
        raise SystemExit(f"{ticker}: the model returned no usable factors")
    with LLM_CORPUS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "ticker": ticker, "model": LLM_MODEL,
                             "weights": weights, "raw": text[:2000]}) + "\n")
    return weights, "recorded"


def fit_and_signal(ticker: str, kind: str) -> tuple[list, dict]:
    from sklearn.preprocessing import StandardScaler

    feats = factor_frame(ticker)
    fwd = price_returns(ticker)
    data = feats.join(fwd, how="inner").replace([np.inf, -np.inf], np.nan)

    cols = [c for c in feats.columns if c.startswith("QLIB158_")]
    train = data[data.index <= TRAIN_END].dropna(subset=cols + ["fwd"])
    test = data[(data.index >= TEST_START) & (data.index <= TEST_END)]

    if len(train) < 500 or test.empty:
        raise SystemExit(f"{ticker}: {len(train)} training rows, {len(test)} test rows")

    # Scaler fit on TRAINING ONLY. Fitting it on the full sample would leak test-window
    # means and variances into the features, which is invisible in the output.
    scaler = StandardScaler().fit(train[cols].to_numpy())
    x = test[cols].to_numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    llm_extra = {}
    if kind == "llm":
        weights, status = llm_select_weights(ticker, train, cols)
        w = np.array([weights.get(c, 0.0) for c in cols])
        pred = scaler.transform(x) @ w
        llm_extra = {"selector_model": LLM_MODEL, "selection": status,
                     "chosen_factors": {k.replace("QLIB158_", ""): v
                                        for k, v in weights.items()},
                     "selector_saw": "training-window rank ICs only; the test window "
                                     "begins after the model's knowledge cutoff"}
    else:
        model = build_model(kind)
        model.fit(scaler.transform(train[cols].to_numpy()), train["fwd"].to_numpy())
        pred = model.predict(scaler.transform(x))

    # Long when the model expects a positive next-day return, cash otherwise: the same
    # long-or-cash convention the committee uses, so the comparison is like for like.
    exposure = (pred > 0).astype(float)
    realised = test["fwd"].to_numpy()

    rows = []
    for date, e, r in zip(test.index, exposure, realised):
        if not np.isfinite(r):
            continue
        rows.append([date, float(e * r), float(e)])
    manifest = {
        "strategy": f"AlphaMining_{kind}",
        "ticker": ticker,
        "returns": "gross",
        "model": kind,
        "features": len(cols),
        "feature_library": "Qlib Alpha158 (mined by Qlib, combined here)",
        "train_rows": int(len(train)),
        "train_end": TRAIN_END,
        "window": [TEST_START, TEST_END],
        "n_scored": len(rows),
        "long_share": float(exposure.mean()),
        "scaler": "fit on training rows only",
        "label": "next-day return, strictly forward",
        "seed": SEED,
    }
    return rows, manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=("ridge", "gbt", "llm", "both"))
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)

    kinds = ["ridge", "gbt"] if args.model == "both" else [args.model]
    names = ([t.strip().upper() for t in args.only.split(",")] if args.only
             else list(TICKERS))
    OUT.mkdir(parents=True, exist_ok=True)

    for kind in kinds:
        print(f"\n=== Alpha158 + {kind} ===")
        print(f"  {'stock':6}{'train':>8}{'scored':>8}{'long%':>8}{'CR%':>9}")
        for ticker in names:
            rows, manifest = fit_and_signal(ticker, kind)
            path = OUT / f"{ticker}__AlphaMining_{kind}.json"
            path.write_text(json.dumps({
                "mode": "baseline",
                "run_label": f"{ticker}__AlphaMining_{kind}",
                "daily_arm_series": {"equal_weight": rows},
                "manifest": manifest,
            }, indent=1), encoding="utf-8")
            cr = float(np.prod([1.0 + r[1] for r in rows]) - 1.0)
            print(f"  {ticker:6}{manifest['train_rows']:8d}{len(rows):8d}"
                  f"{100*manifest['long_share']:7.0f}%{100*cr:9.2f}")
    print("\nscore with: PYTHONPATH=src python scripts/baseline_table.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
