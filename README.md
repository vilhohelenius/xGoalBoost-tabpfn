# xGoalBoost × TabPFN-3.5: NHL expected goals

**Hackathon category:** take on a hard problem (sports analytics).

Can a tabular foundation model predict whether an NHL shot becomes a goal as well as tuned gradient boosting?
We model xG on ~365k shot attempts from [MoneyPuck](https://moneypuck.com/data.htm) (seasons 2023–26, 7.1 % goals)
and compare **TabPFN-3.5 (Prior Labs cloud API)** against XGBoost and CatBoost.

## Results (AUC)

| Model | Train rows | AUC | Log loss |
|---|---|---|---|
| MoneyPuck's own xGoal (reference) | – | 0.782 | – |
| XGBoost | 50 000 | 0.789 | 0.2172 |
| **TabPFN-3.5** | 50 000 | **0.794** | **0.2152** |
| XGBoost | 100 000 | 0.797 | 0.2142 |
| **TabPFN-3.5** | 100 000 | **0.8005** | **0.2128** |
| XGBoost | ~240 000 | 0.802 | 0.2122 |
| XGBoost, leave-one-season-out CV | 2 seasons per fold | 0.806 ± 0.004 | 0.209 |
| CatBoost, leave-one-season-out CV | 2 seasons per fold | 0.805 ± 0.003 | 0.210 |

Setup for the rows down to full-data XGBoost: train on seasons 2023–24, test on a random 30k-shot sample of 2025.
On equal data TabPFN wins at both 50k and 100k rows, and the gap to full-data XGBoost (0.802, ~240k rows) shrinks from
0.008 to 0.001 AUC as TabPFN gets more context. Full-data XGBoost is still marginally best on log loss and AUC;
TabPFN's context size is the limit, so more training rows or ensembling several TabPFN fits is the natural next step.

## A leak we caught

Our first model scored AUC 0.84, better than MoneyPuck's own model, which was suspicious. The cause:
`*Penalty1TimeLeft` / `*Penalty1Length` are recorded *after* the shot. A power-play goal ends the minor penalty, so a
5v4 shot with penalty time 0 is a goal 68 % of the time (vs ~5 %). Removing these columns dropped AUC to a believable
~0.80. See `analysis/penalty_leakage.py`. Also excluded: every MoneyPuck model output (`xGoal`, `x*`),
`shotWasOnGoal`, rebound/play-continued flags, `timeUntilNextEvent`, `homeTeamWon`, score, player and team IDs.

## Run it

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put shots_2023.csv, shots_2024.csv, shots_2025.csv, shots_2026.csv (from moneypuck.com/data.htm) in this folder
.venv/bin/python train.py                       # XGBoost + CatBoost CV (~6 min)
TABPFN_TOKEN=<your key> .venv/bin/python compare_tabpfn.py 50000 30000   # TabPFN vs XGBoost
```

Get a TabPFN token at https://ux.priorlabs.ai/account. `compare_tabpfn.py` runs XGBoost before importing
`tabpfn_client`, because the two crash with a segfault if loaded in the opposite order on macOS.
