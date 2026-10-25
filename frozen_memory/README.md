# Frozen memory

The expert-trust memory that the forward run deployed, one JSON snapshot per stock and memory scope.
It was trained on 2011-07-01 to 2023-12-31, verified once at the freeze, and not changed afterwards.

| File | Holds | Deployed for |
|---|---|---|
| `<SYMBOL>__global.json` | one record per expert over the stock's whole training history | 19 of the 20 stocks |
| `<SYMBOL>__state.json` | one record per (expert, market state) pair | ADSK only |

There is a global snapshot for each of the twenty stocks of the draw, and a state snapshot for seven of
them (ADSK, BIIB, CVX, DHI, ECL, PNC, TGT). A snapshot carries no ticker: it is a trust profile
trained on one stock's own history, so pair each file with its own stock.

## Reading a snapshot

```python
import json
from trading_agent.memory.expert_scorecard import ExpertScorecardStore

snapshot = json.load(open("frozen_memory/AZO__global.json"))
store = ExpertScorecardStore.from_snapshot(snapshot, temperature=0.5)
store.get_weights("global")     # {'fundamental': 0.573, 'news': 0.341, 'technical': 0.087}
```

`python scripts/show_deployed_weights.py` prints the deployed weights for all twenty stocks.

## Fields

| Field | Meaning |
|---|---|
| `cards` | One entry per (`market_state`, `expert_id`): `count` is the number of days the expert spoke, `hits` the number of correct directional views, `ew_mean` and `ew_var` the exponentially weighted mean and variance of its contribution, and `reflection_delta` the bounded adjustment from reflection (at most 0.10 in magnitude) |
| `certified_cells` | The verified records: the (`market_state`, `expert_id`) pairs that passed the verification test. The key keeps the name it had when the files were written; the dissertation calls these verified records. An empty list means nothing passed and the stock deploys equal weights |
| `certified_only` | Only verified records may move a weight away from equal |
| `scope` | `global` (one value of `market_state`) or `state` |
| `status`, `frozen`, `frozen_at` | `frozen`, `true` and `2023-12-31` |
| `half_life`, `min_support`, `variance_penalty`, `standardize`, `temperature` | The retrieval constants: 63 contributions, 20 episodes, 1.0, `true` and 0.5 |
| `hash_sha256` | SHA-256 of the canonical JSON of every other field |

The hash is computed over the fields with the hash removed, serialised with sorted keys and compact
separators:

```python
import hashlib, json

def content_hash(payload):
    body = {k: v for k, v in payload.items() if k != "hash_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
```

`tests/test_frozen_memory.py` recomputes it for every snapshot and pins the deployed weights to the
table that the dissertation fixed before the forward run began.
