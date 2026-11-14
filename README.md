# gpu-pricing-rsi

A team of role-based agents that forecasts GPU rental prices and keeps improving itself through
recursive self-improvement (RSI). The agents share one board of evidence, debate from different
perspectives, and vote with weights that update as results come in. RL on Tinker trains the
agents' weights from the scores their own forecasts earn. Built for Track 2 (Recursive
Self-Improvement) of the iterate.inc AI research hackathon, Nov 14–15, 2026.

> **Status:** design stage. No forecasting code is written yet. The full plan is in
> [docs/PLAN.md](docs/PLAN.md).

## What it forecasts
- **Target:** the daily median price per GPU-hour for H100, H200, B200 and A100, for both
  on-demand and spot rentals.
- **Horizons:** 1, 7 and 30 days ahead.
- **Output:** a range (10th, 50th and 90th percentiles), scored with pinball loss and range coverage.

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
- A test window stays locked until the end and, where possible, starts after the base model's
  training cutoff.
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
.venv/bin/pip install tinker
cp .env.example .env   # add TINKER_API_KEY (file to come with the scaffold)
```

## Open decisions
- **Price data:** collect from public GPU marketplaces plus available history, or use a provided dataset.
- **Track 2 brief:** Pacific's exact question and judging rules are behind the participant login.
- **Base model:** a small Qwen3 or Llama model on Tinker, chosen after checking the model list and prices.
