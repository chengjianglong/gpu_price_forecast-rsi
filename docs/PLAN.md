# gpu-pricing-rsi: working plan

A multi-agent framework that forecasts GPU rental prices and improves itself round by round.
It is built for Track 2 (Recursive Self-Improvement) of the iterate.inc AI research hackathon,
held Nov 14–15, 2026, with judging on Nov 15 at 6 PM. It runs on Tinker only.

## Goal
- **What it forecasts:** the daily median price per GPU-hour for H100, H200, B200 and A100,
  for both on-demand and spot rentals.
- **How far ahead:** 1, 7 and 30 days.
- **What it outputs:** a range (the 10th, 50th and 90th percentiles), so both accuracy and the
  calibration of the uncertainty can be scored.
- **The claim to test:** a team of agents that improves itself beats a statistical baseline, a
  single LLM forecaster, and the same team without self-improvement, on a locked test window.

## 1. Architecture
```
Price data (sliced by date: a call made for date t sees nothing after t)
        │
        ▼
 ┌──────────── Team (one version = one config file) ────────────┐
 │ Data Scout → Quant Analyst ──┐                                │
 │            → Market Analyst ─┴→ Forecasters ×3 → Critic → Aggregator │ → forecast range
 └───────────────────────────────────────────────────────────────┘
        │
        ▼
 Scorer (code only): replays past dates and scores every forecast
        │
        ▼
 Improver: reads where the errors came from and proposes K edited team versions
        │
        ▼
 Selection: keep a version only if it beats the current one on the validation dates
        │
        └── every few rounds: fine-tune the forecaster model on Tinker using its own scored forecasts
```

The team's config file is what improves: the prompts, the model each agent uses, the feature
list and the aggregation weights. Every version is saved with its score, the version it came
from and the reason for the change. That makes the whole history auditable and gives the demo a
family tree of accepted changes.

### Agent roles
| Role | Job | LLM or code |
|---|---|---|
| Data Scout | Collect and clean price series, plus events (GPU launches, cloud price cuts, supply news) | Code; an LLM tags the news |
| Quant Analyst | Statistical baselines (last value, trend and seasonality models, gradient boosting), each with a numeric forecast and range | Code |
| Market Analyst | Read the events and argue for adjustments, e.g. "B200 supply is ramping, so H100 spot prices fall" | LLM |
| Forecasters ×3 | Different starting views: momentum, reverting to normal, supply and demand. Each reads the Quant and Market reports and gives a range | LLM |
| Critic | Challenge each forecast: ranges that are too narrow, reasoning that doesn't hold, any use of information from after the forecast date | LLM |
| Aggregator | Combine the forecasts into the final range, giving more weight to agents with a better recent track record | Code, with an optional LLM |
| Scorer | Replay the past one date at a time and score every forecast | Code only, so agents can't game it |
| Improver | Read the error breakdown and propose one change per candidate: a prompt edit, a feature, a weighting rule or a tool | LLM |

All agents can run on Tinker-hosted models such as Qwen3.

## 2. The self-improvement loop
1. **Backtest:** run the current team over the training dates, using only data available at each date.
2. **Diagnose:** the Scorer breaks the errors down by agent, GPU and horizon, and the Improver
   reads the worst cases.
3. **Propose:** the Improver writes K candidate versions, each with one change. This is a
   population of versions, as in population-based training.
4. **Select:** score each candidate on the validation dates. A candidate is accepted only if it
   lowers pinball loss by more than the noise, measured with a paired bootstrap over forecast dates.
5. **Fine-tune on Tinker:** every few rounds, train the forecaster model on its own scored
   forecasts, with reward = −pinball loss. This is the step where the model itself improves, not
   only its prompts.
6. **Repeat.** The test window stays locked until the final evaluation.

## 3. Repo layout
| Path | What it holds |
|---|---|
| `gpr/data/` | `PriceStore` with an "as of date t" view, the collectors, and the event or news feed |
| `gpr/agents/` | One file per role, all with the same interface: `run(context) -> report` |
| `gpr/team.py` | Runs the agents in order for one forecast date |
| `gpr/llm.py` | Tinker sampling client, plus a fake LLM so tests run offline |
| `gpr/backtest.py` | Splits the dates into train, validation and test windows, and runs the team over them |
| `gpr/score.py` | Pinball loss, mean absolute percentage error, range coverage, and a breakdown by GPU and horizon |
| `gpr/improve.py` | Improver prompts, applying an edit to a config, and the accept-or-reject test |
| `gpr/archive.py` | Every version, its score, its parent and the reason for the change |
| `gpr/finetune.py` | Turns scored forecasts into a Tinker training run |
| `configs/team_v0.yaml` | The starting team |
| `scripts/` | `collect.py`, `backtest.py`, `rsi_loop.py`, `report.py` |
| `tests/` | Leakage tests, scorer tests, and one full round with the fake LLM |

## 4. Rigour
- **No peeking ahead:** every agent call goes through the "as of date t" view, and a test fails
  if any data after t can be read.
- **Locked test window:** scored once, at the end. If possible it starts after the base model's
  training cutoff, so the model can't already know the prices.
- **Fair comparison:** every variant gets the same dates, data and model-call budget.
- **Comparisons, each adding one piece:**
  1. Statistical baseline only
  2. A single LLM forecaster
  3. The full team, without the improvement loop
  4. The team with the improvement loop
  5. The team with the loop plus Tinker fine-tuning
- **Where it breaks:** report the worst forecasts, any change that was accepted but hurt on the
  test window, and the round where improvement stops. A step that adds nothing is reported, not hidden.

## 5. Timeline
| When | Step | Done when |
|---|---|---|
| Nov 14, block 1 (~2 h) | Data and baselines: collectors, `PriceStore`, statistical baselines, scorer | A baseline score on the validation dates |
| Nov 14, block 2 (~3 h) | Agents and team, on the fake LLM and then on Tinker | One forecast date runs end to end on Tinker |
| Nov 14, block 3 (~2 h) | Improver, selection and archive | `rsi_loop.py` finishes 2 rounds |
| Overnight | Run 10–20 rounds and keep collecting live prices | An archive of versions and a score per round |
| Nov 15, morning | One Tinker fine-tuning step, then the comparison runs | The comparison table |
| Nov 15, early afternoon | Report: score-per-round plot, family tree of changes, worst failures | Slides or a page |
| Nov 15, 4 PM | Freeze the results and score the test window once | Final numbers; rehearse before 6 PM judging |

## 6. Risks and fallbacks
| Risk | Fallback |
|---|---|
| Not enough price history | Use only the 1- and 7-day horizons, and score on live prices collected during the event |
| Tinker sampling is too slow for the full team on every date | Backtest on fewer dates, and have one model call play several roles |
| The improvement loop overfits the validation dates | Rotate which dates are used for validation, and rely on the locked test window |
| The model remembers past prices from training | Put the test window after its training cutoff, or use only live-collected data |

## 7. Open decisions
- **Price data:** collect from public GPU marketplaces plus whatever history exists, or use a
  dataset from the Track 2 brief or Pacific.
- **Track 2 brief:** Pacific's exact question, judging rules and any record are on the Resources
  tab, which needs a participant login. Adjust this plan once they are known.
- **Base model on Tinker:** a small Qwen3 or Llama model for the role agents and fine-tuning,
  chosen after checking Tinker's model list and prices.
