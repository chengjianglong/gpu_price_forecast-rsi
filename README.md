# gpu-pricing-rsi

A team of role-based agents that forecasts GPU rental prices and keeps improving itself through
recursive self-improvement (RSI). The agents share one board of evidence, debate from different
perspectives, and vote with weights that update as results come in. RL on Tinker trains the
agents' weights from the scores their own forecasts earn. Built for Track 2 (Recursive
Self-Improvement) of the iterate.inc AI research hackathon, Nov 14–15, 2026.

> **Status:**
> - Done: Pacific's data pack is downloaded and verified, and its starter demo reproduces their
>   baselines locally. The Exa event collector and the time-sliced store are written and tested
>   offline; they haven't been run against the live Exa API yet.
> - Block 2 done: the pack reader, the shared board, the agents (three Forecasters with
>   different lenses, a Critic, a Market Analyst), the debate, the weighted vote, and a script
>   that runs one origin and scores it with Pacific's scorer. It runs offline with a fake LLM
>   and on Tinker (`Qwen/Qwen3.5-9B`).
> - Not written yet: RL, the Improver and the generation loop (block 3).
>
> The full plan is in [docs/PLAN.md](docs/PLAN.md).

## What it forecasts
The task comes from Pacific's GPU price data pack (`gpu-price-pack-v0.1.1-handover`, data
cutoff 6 Nov 2026) and its starter kit.

- **Targets:** the pack's daily series (`series_id` in `series_catalog.csv`), for example AWS spot
  per availability zone, or a collector's median posted rate card for H100 SXM 80GB. The
  headline set depends on the catalog date:
  - **At 24 Oct:** 97 series, 80 of them AWS spot.
  - **At the 6 Nov submission origin:** 37 series, all rental daily statistics with daily truth.
    The AWS archive ends on 31 Oct, so AWS spot falls out of the headline tier.
- **Horizons:** 1, 3, 7 and 30 days from the origin.
- **Output:** `q10, q50, q90` per series and horizon, in the starter kit's forecast CSV layout.
- **Submission:** forecasts from the 6 Nov 2026 origin, which Pacific scores on the withheld
  7–13 Nov truth.
- **Score:** Pacific's `scorer.py` measures pinball loss and reports skill against the last-value
  baseline.

**Baselines to beat**, reproduced locally with the starter demo (pooled skill):

| Baseline | 24 Oct cutoff | 15-origin backtest |
|---|---|---|
| `last_value_empirical` | +0.21 | +0.18 |
| `drift` | +0.20 | +0.17 |

## The agents
| Role | Job | LLM or code |
|---|---|---|
| Data Scout | Collect and clean prices and events (launches, price cuts, supply news) | Code, with an LLM tagging news |
| Quant Analyst | Statistical baseline forecasts with ranges | Code |
| Market Analyst | Argue how events should shift prices | LLM |
| Forecasters ×3 | Forecast from different views: momentum, reverting to normal, supply and demand | LLM |
| Critic | Challenge weak reasoning, ranges that are too narrow, and any look-ahead | LLM |
| Aggregator | Weighted vote over the revised forecasts | Code |
| Scorer | Replay past dates and score every forecast | Code only |
| Improver | Propose changes to prompts, features and the weighting rule | LLM |

## How one forecast is made
1. **Shared board:** for each date, every agent reads the same evidence pack (prices, baselines
   and events up to that date, each with an ID), the agents' track records, the current voting
   weights, and lessons from past mistakes.
2. **Independent round:** each Forecaster posts a position citing evidence IDs, without seeing
   the others' positions.
3. **Debate:** the Critic and the Forecasters challenge each other's positions (at most 2 rounds),
   and each Forecaster revises or defends its forecast.
4. **Vote:** the Aggregator combines the revised forecasts with weights that depend on GPU,
   horizon and market regime. The more the Forecasters still disagree, the wider the range.

## How it improves itself
| Level | What changes | How |
|---|---|---|
| Agent weights | Each LLM role's LoRA adapter on Tinker | GRPO-style RL: sample several answers per date, reward each by its score against the real price, update with `forward_backward` + `optim_step` |
| Voting weights, fast | Each agent's weight per context | Hedge update after every date whose outcome is known, with decay towards equal weights and a minimum weight |
| Voting weights, slow | The weighting rule's settings | Varied by population training and kept only if they beat the current rule |
| The team | Prompts, features, tools | Improver edits, kept only if they pass the acceptance test |
| Generations | The whole population | Population-based training: the best member is copied and varied, then distilled into a fresh starting point |

Rewards are computed by code, never by another LLM. The loop stops when the validation score
hasn't improved beyond noise for 3 generations, when training and validation scores diverge, or
when the time or compute budget runs out.

## Rigour
- Agents see only data available on the forecast date, and tests fail on any leak.
- A locked test at the 24 Oct origin is scored once at the end, in Pacific's frozen-cutoff mode.
- The 6 Nov submission is checked with Pacific's scorer in frozen-cutoff mode before it is sent.
- All four of the scorer's skill aggregates are reported, plus range coverage.
- Comparisons add one piece at a time:
  - the statistical baseline;
  - a single LLM forecaster;
  - the team without and with debate;
  - equal versus RSI-updated voting weights;
  - Improver edits only;
  - RL;
  - the full loop.

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # add EXA_API_KEY (from the Exa dashboard) and TINKER_API_KEY
.venv/bin/python -m pytest -q
```

## Data
### Pacific's data pack (main source)
Get it from the Track 2 Google Drive folder. Then check and unpack it into `data/raw/`, which is
git-ignored:

```bash
shasum -a 256 data/raw/gpu-price-pack-v0.1.1-handover_pack-only.zip   # compare with ZIP.sha256
cd data/raw && unzip -q gpu-price-pack-v0.1.1-handover_pack-only.zip
cd gpu-price-pack-v0.1.1-handover && shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$'   # no output = intact
```

Reproduce Pacific's baselines with the starter demo (about 20 s, offline):

```bash
.venv/bin/pip install duckdb pandas numpy pyarrow
PYTHON="$PWD/.venv/bin/python" bash data/raw/gpu-price-pack-v0.1.1-handover/starter/run_demo.sh \
    "$PWD/data/raw/gpu-price-pack-v0.1.1-handover" "$PWD/data/derived/demo" 2026-10-24
```

Rules from the pack's `DATA.md`:
- Every price in it is a posted price, not a price paid.
- Instrument classes are never chained into one series.
- Keep `ATTRIBUTION.md` with any derived work.
- Only add external data that is public, free to every team and dated before the origin.

### Exa (news events, supplementary)
`gpr/data/` also collects from [Exa](https://exa.ai). Its main use is news events, dated before
each origin, for the shared board. Whether Exa meets the pack's external-data rule still needs
confirming with the organisers. It can collect:
- **History:** stored snapshots of provider pricing pages at past dates (`snapshot_as_of`),
  timestamped by the snapshot's own crawl time.
- **Live:** the same pages fetched fresh, run on a schedule during the event.
- **Articles:** prices quoted in news and blogs, used as evidence rather than as the target.
- **Events:** news found with published-date filters, which goes into the shared board.

Prices are extracted with Exa's structured summaries, converted to USD per GPU-hour, and checked
against plausible bounds. `PriceStore.as_of(t)` only returns what was knowable at `t`.

```bash
.venv/bin/python scripts/collect.py discover                     # find pricing pages
.venv/bin/python scripts/collect.py history --start 2025-05-01 --every-days 7 --dry-run
.venv/bin/python scripts/collect.py history --start 2025-05-01 --every-days 7
.venv/bin/python scripts/collect.py events --start 2025-05-01 --every-days 30
.venv/bin/python scripts/collect.py live                         # repeat during the event
.venv/bin/python scripts/collect.py summary
```

## Run one origin
```bash
.venv/bin/python scripts/run_origin.py --origin 2026-10-15 --llm fake              # offline dry run
.venv/bin/python scripts/run_origin.py --origin 2026-10-15 --llm tinker --limit 3  # needs TINKER_API_KEY in .env
```
It writes `outputs/<run>/forecast_<origin>.csv` in Pacific's forecast format, plus the debate
transcript and the scorer's report. The 24 Oct and 6 Nov origins are checked in frozen-cutoff
mode. Series are chosen from the catalog dated at the origin, as the scorer requires.

## Open decisions
- **Exa:** confirm it meets the pack's external-data rule; how far back its page snapshots go is
  unknown until the first run.
- **Model knowledge:** several Tinker models look like 2026 releases, so their training data may
  overlap the pack's 2026 prices. Check each model's cutoff, and test recall without the data,
  before trusting backtest scores.
- **Track 2 brief:** the submission steps and which skill aggregate ranks teams are in the full
  brief, behind the participant login.
- **Base model:** a small Qwen3 or Llama model on Tinker, chosen after checking the model list and prices.
