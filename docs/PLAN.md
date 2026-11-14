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
 RL on Tinker: each LLM agent's weights are trained on the rewards its own answers earned
        │
        ▼
 Selection: population members compete on the validation dates; the best is copied and varied
        │
        └── repeat until validation performance stops improving (section 2.4)
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

### Shared information and debate
The agents work from the same facts so they can argue about what those facts mean, not about
what the facts are.

**The shared board.** For each forecast date t there is one board that every agent reads.
| Section | What it holds | Written by |
|---|---|---|
| Evidence pack | Price series up to t, the baseline forecasts with their ranges, and the event timeline up to t. Every item has an ID, e.g. `E12` | Data Scout, Quant Analyst |
| Track record | Each agent's recent pinball loss and range coverage, by GPU and horizon | Scorer |
| Lessons | Short post-mortems of past mistakes | Improver |
| Positions | Each agent's forecast, its reasoning, and the evidence IDs it relies on | Forecasters, Market Analyst |
| Challenges | Objections, each aimed at one position and citing evidence | Critic, Forecasters |

**Debate protocol for one date:**
1. **Independent round:** each Forecaster reads the board and posts a position without seeing
   the others' positions, so they don't anchor on each other.
2. **Challenge round:** the Critic and the Forecasters post challenges to each other's positions,
   for example "E12 shows spot prices fell 9% in a week; your median ignores it".
3. **Revision round:** each Forecaster answers the challenges it received and may revise its
   forecast. It must say what changed its mind, or why it holds its position.
4. **Decision:** the Aggregator combines the revised forecasts. The more the Forecasters still
   disagree, the wider the final range.

The debate lasts at most 2 challenge-and-revision rounds, to keep the cost per date bounded.

**Rules that keep the debate useful:**
- **Cite evidence:** every claim must cite at least one evidence ID. The reward code flags claims
  without citations or with IDs that don't exist.
- **Different perspectives:** every agent sees the same evidence pack, but each Forecaster keeps
  its own lens (momentum, reverting to normal, supply and demand). If they converge, it should
  be because the evidence persuaded them, not because they all share a prompt.
- **Guard against everyone agreeing:** the Forecasters are scored on their own revised forecasts,
  so nobody gains by copying the majority. A Forecaster that changes its forecast in the right
  direction after a valid challenge earns a bonus.
- **No leakage through lessons:** a lesson about date d can only appear on boards for dates after
  d's outcome is known.

**How the debate feeds the RL** (section 2.1):
- Forecasters are rewarded on their revised forecasts, plus the bonus for useful revisions.
- The Critic is rewarded for challenges that lead to better revised forecasts.
- The whole transcript is kept with each RL example, so the training data includes how the
  debate went, not only the final answer.

## 2. Recursive self-improvement with RL
Improvement happens at two levels, and the loop keeps running until validation performance
stops rising.

- **Inner level, the weights:** RL on Tinker trains the LLM agents' weights from the scores
  their own forecasts earn.
- **Outer level, the team:** each generation, a population of trained versions competes. The
  best is copied and varied, the Improver edits prompts and features, and the next generation
  starts from the winner.

### 2.1 Inner level: RL on Tinker
Each trainable role has its own LoRA adapter on one shared base model (one Tinker training
client per role).

**One RL step for a role:**
1. Draw a batch of training dates t, giving extra weight to dates the team got most wrong
   (the curriculum in 2.3).
2. Build each prompt from data as of t only.
3. Sample G answers per prompt from the role's current sampling client.
4. The Scorer computes each answer's reward against the price that actually happened.
5. Advantage = reward − the mean reward of its group of G (GRPO-style: no value model needed).
6. Call `forward_backward` with an RL loss (`importance_sampling` to start, `ppo` if updates
   are unstable), then `optim_step`.
7. Call `save_weights_and_get_sampling_client` so the next step samples from the updated weights.

**Rewards** (computed by code only; no LLM judges an LLM):
| Role | Reward |
|---|---|
| Forecasters | Skill against the statistical baseline: `1 − pinball(forecast) / pinball(baseline)`, minus penalties for crossed quantiles or invalid output |
| Critic | How much the forecasters' pinball loss drops after they revise using its critique (critique that doesn't help scores 0) |
| Market Analyst | Leave-one-out contribution: the team's pinball loss without its report minus the loss with it, measured on a sample of dates because it costs an extra team run |
| Aggregator | Not RL: its voting weights are updated by the rule in 2.6, and the rule itself evolves across generations |
| Data Scout and Quant Analyst | Code; they improve through the Improver's edits in 2.2 |

**Guards against reward hacking and collapse:**
- A KL penalty keeps each adapter close to its starting point.
- Watch how varied the sampled answers are; if the G answers become identical, raise the
  temperature or the KL weight.
- The reward code checks the output format and penalises crossed quantiles.
- Leakage tests run on every prompt the RL step builds.

### 2.2 Outer level: generations
Each generation does the following:
1. **Train:** run the inner RL for N steps on each of K population members. Members differ in
   learning rate, KL weight, temperature and group size.
2. **Evaluate:** score every member's full team on the validation dates.
3. **Exploit and explore** (population-based training, as in the earlier repo): the bottom
   members copy the best member's adapters and then vary its settings.
4. **Edit:** the Improver reads the error breakdown and proposes changes to prompts, features,
   tools and the Aggregator rule. A change is kept only if it passes the acceptance test.
5. **Distil** (every few generations): fine-tune a fresh adapter with supervised learning
   (`cross_entropy`) on the best member's highest-reward answers. This starts the next round of
   RL from a clean, strong point.
6. **Archive:** save every member, its score, its parent and its settings.

**Acceptance test:** a new best must lower validation pinball loss by more than the noise,
measured with a paired bootstrap over forecast dates.

### 2.3 Curriculum
- **Hard-date replay:** sample training dates in proportion to the current team's error on them.
- **Horizons:** start with the 1-day horizon, add 7 days, then 30 days as the shorter ones stop
  improving.
- **Regime breaks:** keep the large price moves (launches, price cuts) over-represented, because
  they are where the team fails most.

### 2.4 When to stop
The loop continues while there is real improvement, and stops at the first of these:
- **Plateau:** validation skill has not improved beyond the bootstrap noise for P = 3 generations in a row.
- **Overfitting:** the training score keeps rising while the validation score falls for 2 generations.
- **Budget:** the Tinker compute budget or the time box (results frozen at 4 PM on Nov 15) runs out.

"As high as possible" has a real ceiling: prices have noise no forecaster can predict. To
estimate how close we are, the report includes a noise floor: the pinball loss of the best model
fitted with hindsight on the test window. A gap that stops shrinking is the stopping signal, not
a failure.

### 2.6 Voting weights
The Aggregator combines the revised forecasts as a weighted vote, separately for each quantile:
`final_q = Σ_i w_i(context) · forecast_i,q`. The weights keep updating as the loop runs, at
two speeds.

**Fast: after every date whose outcome is known.**
- Each agent's weight moves with its recent pinball loss, using exponentiated gradient (Hedge):
  `w_i ← w_i · exp(−η · loss_i)`, then normalised.
- **Context:** separate weights for each GPU × horizon × market regime (calm, trending, shock),
  because an agent that is good at 1-day spot prices may be poor at 30-day on-demand.
- **Stability:** weights decay towards equal weights over time (`decay`), shrink towards equal
  weights in contexts with few outcomes, and never fall below a floor (`w_min`). The floor keeps
  every perspective in the vote, so a currently out-of-favour agent can win back weight when
  the regime changes.
- **Debate signals:** an agent's weight can also depend on how strong its position was in the
  debate: whether it held up under challenge, and how well its evidence citations checked out.

**Slow: the rule itself improves, once per generation.**
- The rule's settings are part of each population member's config: `η`, `decay`, `w_min`, the
  context buckets, and how much the debate signals count.
- Population training (2.2) varies these settings, and the Improver can propose new weighting
  features.
- A new rule is kept only if it passes the same acceptance test on the validation dates.
- **Optional, if time allows:** learn the weights directly as a small gating model that maps
  context to weights, trained to minimise the combined pinball loss on training dates. Because
  the combined loss is convex in the weights, this is a cheap optimisation, not RL.

**Feedback to the agents:** the current weights appear on the shared board's track record, so
every agent sees whose views count most in which contexts.

**No leakage:** the weights for date t use only outcomes known before t.

**Reported:** how each agent's weight changes over time and by context, plus comparison 3c in
section 4.

### 2.7 Data volume
RL needs many training examples, and daily price history is short. Every GPU × provider ×
on-demand/spot × horizon combination is a separate example, and hourly prices multiply that
again. With little data, overfitting is the main risk, so the stopping rules and the locked
test window matter more than usual.

## 3. Repo layout
| Path | What it holds |
|---|---|
| `gpr/data/` | `PriceStore` with an "as of date t" view, the collectors, and the event or news feed |
| `gpr/agents/` | One file per role, all with the same interface: `run(context) -> report` |
| `gpr/board.py` | The shared board: evidence pack with IDs, track record, lessons, positions and challenges |
| `gpr/debate.py` | The debate protocol: independent round, challenge rounds, revision rounds, decision |
| `gpr/team.py` | Runs the agents and the debate for one forecast date |
| `gpr/llm.py` | Tinker sampling client, plus a fake LLM so tests run offline |
| `gpr/backtest.py` | Splits the dates into train, validation and test windows, and runs the team over them |
| `gpr/score.py` | Pinball loss, mean absolute percentage error, range coverage, and a breakdown by GPU and horizon |
| `gpr/improve.py` | Improver prompts, applying an edit to a config, and the accept-or-reject test |
| `gpr/archive.py` | Every version, its score, its parent and the reason for the change |
| `gpr/weights.py` | Voting weights: context buckets, the Hedge update, decay, floor, and the optional gating model |
| `gpr/rewards.py` | Per-role rewards, format checks and the quantile-crossing penalty |
| `gpr/rl.py` | One RL step per role on Tinker: sample G answers, score them, compute advantages, `forward_backward`, `optim_step` |
| `gpr/population.py` | Population members, exploit and explore, distillation, and the stopping rules |
| `gpr/curriculum.py` | Hard-date replay, horizon schedule and over-sampling of regime breaks |
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
  3. The full team, without the improvement loop (3a: no debate, the Forecasters are simply
     averaged; 3b: with the debate), to show what the debate adds. 3c compares voting
     weights: equal weights, weights fitted once on the training dates, and weights updated by
     RSI (2.6)
  4. The team with the Improver's edits only (no weight training)
  5. The team with RL on the agents' weights
  6. The full loop: RL, population training, Improver edits and distillation
- **Curves:** validation skill per generation for each population member, with the stopping
  point marked, and the training-validation gap.
- **Where it breaks:** report the worst forecasts, any change that was accepted but hurt on the
  test window, and the round where improvement stops. A step that adds nothing is reported, not hidden.

## 5. Timeline
| When | Step | Done when |
|---|---|---|
| Nov 14, block 1 (~2 h) | Data and baselines: collectors, `PriceStore`, statistical baselines, scorer | A baseline score on the validation dates |
| Nov 14, block 2 (~3 h) | Agents and team, on the fake LLM and then on Tinker | One forecast date runs end to end on Tinker |
| Nov 14, block 3 (~3 h) | Rewards, one RL step per role on Tinker, Improver, selection and archive | One generation runs end to end: rewards rise on the training dates |
| Overnight | Run the full loop (population of K = 4) until a stopping rule fires; keep collecting live prices | An archive of generations and a validation curve |
| Nov 15, morning | Comparison runs (the six rows in section 4), resumed from the overnight checkpoints | The comparison table |
| Nov 15, early afternoon | Report: score-per-round plot, family tree of changes, worst failures | Slides or a page |
| Nov 15, 4 PM | Freeze the results and score the test window once | Final numbers; rehearse before 6 PM judging |

## 6. Risks and fallbacks
| Risk | Fallback |
|---|---|
| Not enough price history | Use only the 1- and 7-day horizons, and score on live prices collected during the event |
| Tinker sampling is too slow for the full team on every date | Backtest on fewer dates, and have one model call play several roles |
| The improvement loop overfits the validation dates | Rotate which dates are used for validation, and rely on the locked test window |
| The model remembers past prices from training | Put the test window after its training cutoff, or use only live-collected data |
| RL is unstable or the answers collapse to one | Switch the loss from `importance_sampling` to `ppo`, raise the KL weight, or restart from the last distilled adapter |
| RL overfits the few training dates | Fewer RL steps per generation, more series per date, and the overfitting stop rule |
| The debate multiplies model calls per date | One challenge round during RL training, two in evaluation; debate only on dates where the Forecasters disagree by more than a threshold |
| RL is too slow on Tinker overnight | Train only the Forecasters with RL and leave the other roles to the Improver |

## 7. Open decisions
- **Price data:** collect from public GPU marketplaces plus whatever history exists, or use a
  dataset from the Track 2 brief or Pacific.
- **Track 2 brief:** Pacific's exact question, judging rules and any record are on the Resources
  tab, which needs a participant login. Adjust this plan once they are known.
- **Base model on Tinker:** a small Qwen3 or Llama model for the role agents and fine-tuning,
  chosen after checking Tinker's model list and prices.
