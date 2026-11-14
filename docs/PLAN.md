# RSI4GPF (gpu_price_forecast-rsi): working plan

A multi-agent framework that forecasts GPU rental prices and improves itself round by round.
It is built for Track 2 (Recursive Self-Improvement) of the iterate.inc AI research hackathon,
held Nov 14–15, 2026, with judging on Nov 15 at 6 PM. It runs on Tinker only.

## Goal
The task is set by Pacific's data pack (`gpu-price-pack-v0.1.1-handover`) and its starter kit.

- **What it forecasts:** the pack's daily price series, identified by the `series_id` values in
  `series_catalog.csv`. In 2026 there are 725 series, 97 of them in the headline tier. Examples:
  - `aws_spot.p5.48xlarge.use1-az5`: AWS spot price per GPU-hour, for one availability zone.
  - `gpurentalprices.neocloud_rate_card.H100_SXM_80GB.on_demand.median`: the daily median
    posted rate from one price collector.
- **How far ahead:** 1, 3, 7 and 30 days from the origin date.
- **What it outputs:** `q10, q50, q90` per series and horizon, in the CSV layout of
  `starter/FORECAST_SCHEMA.md`.
- **The submission:** forecasts made from the origin of **6 Nov 2026** (the pack's data cutoff).
  Pacific scores them against truth for 7–13 Nov, which is withheld from the pack.
- **The score:** Pacific's `scorer.py` computes pinball loss at quantiles 0.1, 0.5 and 0.9, and
  reports skill = 1 − loss / loss of the last-value baseline. It reports four aggregates:
  pooled, mean of series, median of series and geometric mean. The organisers choose which one
  ranks teams.
- **The bar to beat:** on the starter demo, Pacific's best baselines reach about +0.20 pooled
  skill at the 24 Oct cutoff (`last_value_empirical` +0.21, `drift` +0.20). Over the 15-origin
  backtest from 15 Jul to 24 Oct they reach about +0.17 to +0.18. We reproduced these numbers
  locally.
- **The claim to test:** a team of agents that improves itself beats Pacific's baselines, a
  single LLM forecaster, and the same team without self-improvement.

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
| Data Scout | Build each series' history up to the origin from the pack (`starter/build_series.py --cutoff`), plus related features from the pack's other tables (p25/p75/minimum, provider counts, indices, aggregator medians, hardware and capability data), plus news events from Exa dated before the origin | Code; an LLM tags the news |
| Quant Analyst | Pacific's four baselines (`starter/baselines.py`: `last_value`, `last_value_empirical`, `drift`, `seasonal_naive`), plus our own statistical models, each with q10/q50/q90 | Code |
| Market Analyst | Read the events and argue for adjustments, e.g. "B200 supply is ramping, so H100 spot prices fall" | LLM |
| Forecasters ×3 | Different starting views: momentum, reverting to normal, supply and demand. Each reads the Quant and Market reports and gives a range | LLM |
| Critic | Challenge each forecast: ranges that are too narrow, reasoning that doesn't hold, any use of information from after the forecast date | LLM |
| Aggregator | Combine the forecasts into the final range, giving more weight to agents with a better recent track record | Code, with an optional LLM |
| Scorer | Pacific's `starter/scorer.py`, run in backtest mode during training and in frozen-cutoff mode for the submission, so our numbers match the official scoring | Code only, so agents can't game it |
| Improver | Read the error breakdown and propose one change per candidate: a prompt edit, a feature, a weighting rule or a tool | LLM |

All agents can run on Tinker-hosted models such as Qwen3.

### Shared information and debate
The agents work from the same facts so they can argue about what those facts mean, not about
what the facts are.

**The shared board.** For each forecast date t there is one board that every agent reads.
| Section | What it holds | Written by |
|---|---|---|
| Evidence pack | Each series' history up to t with its instrument class and benchmark contract, the baseline forecasts with their ranges, related pack tables, and the event timeline up to t. Every item has an ID, e.g. `E12` | Data Scout, Quant Analyst |
| Track record | Each agent's recent pinball loss and range coverage, by instrument class and horizon | Scorer |
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

> **As built in block 3** (`gpr/rl.py`, `gpr/loop.py`, `gpr/evaluate.py`, `gpr/improve.py`).
> These are the differences from the design below, made to fit the time left:
> - **One shared adapter for the Forecasters:** the three lenses share one LoRA adapter, with
>   the lens in the prompt. The Critic and Market Analyst use the same adapter but are not
>   trained with RL yet.
> - **No KL penalty yet:** a small learning rate and the plateau and overfitting stop rules
>   limit drift instead.
> - **Distillation deferred:** generations restart weaker members from the best member's
>   saved state instead.
> - **Curriculum:** examples are drawn by price move size and the adapter's last reward on
>   them (hard-date replay). There is no separate horizon schedule yet.
> - **Truth caps:**
>   - RL rewards use truth dated on or before 30 Sep, the day before the first validation
>     origin.
>   - Validation uses truth dated on or before 24 Oct, the locked test origin.
> - **Generation 0's incumbent** is Pacific's `last_value_empirical` baseline. A member
>   replaces the incumbent only when a paired bootstrap gives `p_better >= 0.9`.
> - **After checking the full brief (14 Nov):**
>   - Train, validation and test use the posted rental series with daily truth
>     (`gpr.pack.rental_scope`), not AWS spot.
>   - Rewards and scores use only the scored horizons (1, 3, 7).
>   - The locked test is three origins (24, 27, 30 Oct).
>   - A research notebook (`gpr/research.py`) proposes hypotheses about price drivers, tests
>     them on the training window, and posts survivors on the boards.
>   - `scripts/final.py` runs the test with ablations and writes the submission.
>   - `scripts/long_run.py` builds the 2–3 year view.
>   - `reproduce.sh` is the one command.
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
| Forecasters | Skill per cell, as Pacific's scorer defines it: `1 − pinball(forecast) / pinball(last_value)`, clipped to a fixed range so near-flat series can't dominate, minus penalties for rows the schema would reject (crossed quantiles, non-positive values) |
| Critic | How much the forecasters' pinball loss drops after they revise using its critique (critique that doesn't help scores 0) |
| Market Analyst | Leave-one-out contribution: the team's pinball loss without its report minus the loss with it, measured on a sample of dates because it costs an extra team run |
| Aggregator | Not RL: its voting weights are updated by the rule in 2.5, and the rule itself evolves across generations |
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
- **Horizons:** start with the 1-day horizon, then add 3, 7 and 30 days as the shorter ones stop
  improving.
- **Instrument classes:** train on AWS spot first, because it has the longest history (May 2022
  onwards). Then add the rental daily statistics, which the 7–13 Nov scoring relies on because
  AWS spot has no November truth.
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

### 2.5 Voting weights
The Aggregator combines the revised forecasts as a weighted vote, separately for each quantile:
`final_q = Σ_i w_i(context) · forecast_i,q`. The weights keep updating as the loop runs, at
two speeds.

**Fast: after every date whose outcome is known.**
- Each agent's weight moves with its recent pinball loss, using exponentiated gradient (Hedge):
  `w_i ← w_i · exp(−η · loss_i)`, then normalised.
- **Context:** separate weights for each instrument class × horizon × market regime (calm,
  trending, shock), because an agent that is good at 1-day spot prices may be poor at 30-day
  rate cards.
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

### 2.6 Data and splits
**Volume.** The pack is large enough for RL:
- AWS spot change events from May 2022 to Oct 2026 (31 million rows, with Mar–Jun 2026 missing
  and never filled in).
- Rental daily statistics from 12 Jan to 6 Nov 2026.

Every (series, origin, horizon) is one training example. Because posted rate cards move in steps,
many rental series barely move, so the curriculum over-weights the ones that do.

**Splits by origin date:**
| Split | Origins | Use |
|---|---|---|
| Train | up to 30 Sep 2026 | RL and Improver diagnostics |
| Validation | weekly, 1 Oct to 17 Oct 2026 | Acceptance tests and stopping rules |
| Locked test | 24 Oct 2026, frozen mode, truth up to 6 Nov | Scored once at the end, like Pacific's demo |
| Submission | 6 Nov 2026, frozen mode | Pacific scores it on the withheld 7–13 Nov truth |

**Headline set by origin.** The catalog's tier gates depend on the date. At 24 Oct the headline
set is 97 series (80 AWS spot). At 6 Nov it is 37 series, all rental daily statistics with daily
truth, because the AWS archive ends on 31 Oct. Every origin chooses its series from the catalog
dated at that origin (`gpr.pack.series_at`), as Pacific's scorer does.

**Model knowledge.** Several Tinker models look like 2026 releases, so their training data may
overlap the pack's 2026 prices. Before trusting backtest scores, check each candidate model's
training cutoff and test whether it can recall prices without the evidence. If it can, keep
validation and test origins after the cutoff.

**Rules from the pack (DATA.md sections 3 and 9):**
- Never chain different instrument classes into one series.
- Quote each series' instrument class and benchmark contract with any result.
- Keep `ATTRIBUTION.md` with our work.
- Add only external data that is public, free to every team and dated before the origin. No
  Vast.ai, RunPod or the other barred sources.

## 3. Repo layout
| Path | What it holds |
|---|---|
| `gpr/data/` | `PriceStore` with an "as of date t" view, the Exa collector for news events, and an adapter that reads the pack's series files and catalog |
| `data/raw/` (git-ignored) | Pacific's pack, unpacked and checked against its manifest |
| `gpr/agents/` | One file per role, all with the same interface: `run(context) -> report` |
| `gpr/board.py` | The shared board: evidence pack with IDs, track record, lessons, positions and challenges |
| `gpr/debate.py` | The debate protocol: independent round, challenge rounds, revision rounds, decision |
| `gpr/team.py` | Runs the agents and the debate for one forecast date |
| `gpr/llm.py` | Tinker sampling client, plus a fake LLM so tests run offline |
| `gpr/backtest.py` | Splits the dates into train, validation and test windows, and runs the team over them |
| `gpr/score.py` | Runs Pacific's `scorer.py` and reads its JSON report and per-cell CSV, adding breakdowns by instrument class, horizon and agent |
| `gpr/improve.py` | Improver prompts, applying an edit to a config, and the accept-or-reject test |
| `gpr/archive.py` | Every version, its score, its parent and the reason for the change |
| `gpr/weights.py` | Voting weights: context buckets, the Hedge update, decay, floor, and the optional gating model |
| `gpr/rewards.py` | Per-role rewards, format checks and the quantile-crossing penalty |
| `gpr/rl.py` | One RL step per role on Tinker: sample G answers, score them, compute advantages, `forward_backward`, `optim_step` |
| `gpr/population.py` | Population members, exploit and explore, distillation, and the stopping rules |
| `gpr/curriculum.py` | Hard-date replay, horizon schedule and over-sampling of regime breaks |
| `configs/team_v0.yaml` | The starting team |
| `scripts/` | `collect.py` (Exa), `backtest.py`, `rsi_loop.py`, `submit.py` (writes and validates the 6 Nov forecast file in frozen mode), `report.py` |
| `tests/` | Leakage tests, scorer tests, and one full round with the fake LLM |

## 4. Rigour
- **No peeking ahead:** every agent call goes through the "as of date t" view, and a test fails
  if any data after t can be read.
- **Locked test window:** the 24 Oct frozen-mode test in 2.6 is scored once, at the end.
- **Frozen mode:** the submission is checked with Pacific's scorer in frozen-cutoff mode before
  it is sent. The scorer refuses any origin other than the cutoff, or a history file with rows
  after it.
- **All four aggregates:** report pooled, mean-of-series, median-of-series and geometric-mean
  skill, plus range coverage. Mean-of-series skill is negative even for Pacific's best baselines
  because a few near-flat series dominate it, so no single aggregate is cherry-picked.
- **Fair comparison:** every variant gets the same dates, data and model-call budget.
- **Comparisons, each adding one piece:**
  1. Pacific's baselines (`last_value_empirical`, `drift`)
  2. A single LLM forecaster
  3. The full team, without the improvement loop (3a: no debate, the Forecasters are simply
     averaged; 3b: with the debate), to show what the debate adds. 3c compares voting
     weights: equal weights, weights fitted once on the training dates, and weights updated by
     RSI (2.5)
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
| Nov 14, block 1 | **Done:** Pacific's pack downloaded and verified; starter demo reproduces their baselines; Exa event collector and `PriceStore` written and tested offline | Baseline scores at the 24 Oct cutoff and on the 15-origin backtest |
| Nov 14, block 2 | **Done:** pack adapter and splits; agents, shared board and debate, on the fake LLM and then on Tinker | One origin runs end to end on Tinker and writes a valid forecast file |
| Nov 14, block 3 (~3 h) | Rewards, one RL step per role on Tinker, Improver, voting weights, selection and archive | One generation runs end to end: rewards rise on the training origins |
| Overnight | Run the full loop (population of K = 4) until a stopping rule fires | An archive of generations and a validation curve |
| Nov 15, morning | Comparison runs (the six rows in section 4), resumed from the overnight checkpoints | The comparison table |
| Nov 15, early afternoon | Locked 24 Oct test, scored once; the 6 Nov submission file, checked in frozen mode; report with score-per-generation plot, family tree of changes and worst failures | Final numbers and the forecast file |
| Nov 15, 4 PM | Freeze the results | Rehearse before 6 PM judging |

## 6. Risks and fallbacks
| Risk | Fallback |
|---|---|
| Rental series barely move, so last value is hard to beat | Lean on calibrated ranges (most skill comes from q10/q90), and over-weight the series that move in the curriculum |
| AWS spot has no truth for 7–13 Nov | Use it for training and validation only; the submission's skill comes from the rental series |
| Exa may not count as free to every team | Check with the organisers; if not, drop Exa events from the submission run and report them as an ablation only |
| Tinker sampling is too slow for the full team on every date | Backtest on fewer dates, and have one model call play several roles |
| The improvement loop overfits the validation dates | Rotate which dates are used for validation, and rely on the locked test window |
| RL is unstable or the answers collapse to one | Switch the loss from `importance_sampling` to `ppo`, raise the KL weight, or restart from the last distilled adapter |
| RL overfits the few training dates | Fewer RL steps per generation, more series per date, and the overfitting stop rule |
| The debate multiplies model calls per date | One challenge round during RL training, two in evaluation; debate only on dates where the Forecasters disagree by more than a threshold |
| RL is too slow on Tinker overnight | Train only the Forecasters with RL and leave the other roles to the Improver |

## 7. Open decisions
- **Track 2 brief:** the full brief (submission steps and which skill aggregate ranks teams) is
  linked from the event's Resources page, which needs a participant login. Adjust this plan once
  it is known.
- **Base model:** `Qwen/Qwen3.5-9B` runs the block-2 team; compare it with a larger model, and
  check its training cutoff first.
- **Exa:** confirm with the organisers that it meets the external-data rule.
