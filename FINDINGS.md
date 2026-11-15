# Findings: a self-improving AI scientist for GPU rental prices

Track 2 (Recursive Self-Improvement by Pacific: An AI Scientist for GPU Value). Direction **A**
(self-improving forecasts), with direction B's memory: a research notebook and a shared evidence
board.

> **Status, 14 Nov 2026:** these are drafted. Sections 1, 2, 3 and 6 are final. Sections 4 and 5
> fill in from `results/report.md` after the overnight run and `scripts/final.py`. Until then
> they hold only the preliminary numbers stated, and claim no gain.

## 1. Question
Two questions, one scored and one argued:

1. **Scored (Sunday):** forecast posted GPU rental prices 1, 3 and 7 days after 6 Nov 2026, with
   calibrated q10/q50/q90. The 37 headline series are all posted rental daily statistics.
2. **Argued (the brief's headline question):** what will H100 and A100 rent for, and resell for,
   in late 2028 and 2029? See section 6.

## 2. Method
A team of LLM agents on Tinker that improves itself at four levels. Two base models,
`Qwen/Qwen3.5-9B` and `DeepSeek-V4.1-Flash`, run identical loops. The better one is chosen on
validation only, by a rule fixed in advance (`scripts/compare_models.py`).

| Level | What changes | How it is kept or rejected |
|---|---|---|
| Research notebook | Tested hypotheses about what moves prices; survivors become citable findings on every board | Permutation or bootstrap test on data up to 30 Sep only; kept if p < 0.05 with the predicted sign |
| Agent weights | One LoRA adapter shared by three Forecasters (momentum, reversion, supply and demand) | GRPO-style RL: reward = skill against the last value at h = 1, 3, 7, on truth up to 30 Sep |
| Voting weights | Each Forecaster's weight per instrument class and horizon | Hedge update after each outcome known before the next origin; the rule's settings evolve |
| Team config | Prompts, debate rounds, the voting rule | Population training plus Improver edits; kept only if a paired bootstrap on validation gives P(better) >= 0.9 |

**One forecast:**
1. Every agent reads the same board: the evidence pack, findings and track record.
2. The Forecasters post independently.
3. The Critic challenges them, citing evidence IDs.
4. The Forecasters revise.
5. A weighted vote decides, and remaining disagreement widens the range.

Pacific's own `baselines.py` supplies the statistical evidence, and its `scorer.py` does all the
scoring.

**A no-LLM yardstick, ported from the sister project `gpu-price`:** the list-replay engine
(`gpr/engine.py`). Every posted rental series is the median of one collector's list of offers.
Drift replays how the median moved. The engine replays what each offer did over past pairs of
days (repriced, stayed, left, joined), applies that to today's list and takes the median again.
Nothing is trained. It runs as a comparison arm in the locked test. A second team config
(`configs/team_list.yaml`) also puts the offer list and the engine's forecast on the board and
gives the Forecasters list-reading jobs; it is a candidate for the loop, not the default.

## 3. Evaluation design (fixed before looking)
| Split | Origins | Truth used | Purpose |
|---|---|---|---|
| Train | weekly, 4 Feb to 30 Sep 2026 | up to 30 Sep | RL rewards and the research tests |
| Validation | 1, 8, 15 Oct | up to 24 Oct | acceptance and stopping |
| **Locked test** | **24, 27, 30 Oct** | up to 6 Nov, frozen-cutoff mode | scored once, on origins no change ever saw |
| Submission | 6 Nov | withheld 7–13 Nov (Pacific) | Sunday |

Scope throughout: posted rental series with daily truth and at least 30 days of history at the
origin, chosen from the catalog dated at that origin, so there is no look-ahead in which series
are used. AWS spot is excluded: it has no November truth and is a different instrument. The
Sunday 37-series subset is reported separately.

Comparisons on the same test cells:
- the trained team;
- the trained team without research findings;
- the untrained team;
- Pacific's `drift` and `last_value_empirical` baselines;
- the list-replay engine. This arm is not blind: the sister project had scored the engine on
  forecast days from 16 Oct to 5 Nov, which include these origins, before it was ported.

Each comes with a 90% interval and a paired bootstrap against `drift`.

## 4. Results
*Fills from `results/report.md`.* Preliminary numbers, before RL training:
- Pacific's `last_value_empirical` on the validation rental scope: pooled skill **+0.162**.
- The untrained team at 15 Oct, on the earlier 97-series headline scope (mostly AWS spot):
  **+0.218**, level with the baseline (+0.217). No gain is claimed from that.
- Untrained teams put 23–24% of outcomes above q90 (target 10%): their ranges are too narrow on
  the upside.

**The list-replay engine** (final; `scripts/engine_check.py`, full tables in
[results/engine.md](results/engine.md)). Rental scope, h = 1, 3, 7, Pacific's scorer. The locked
test is not touched.

| Window | Cells | Engine pooled skill | Drift pooled skill | Engine against drift (less error) |
|---|---|---|---|---|
| Training window: 24 weekly origins, truth to 30 Sep | 6,118 | +0.121 | +0.083 | +4.1% (+1.3% to +6.9%) |
| Validation: 1, 8, 15 Oct | 2,458 | +0.171 | +0.165 | +0.7% (−6.2% to +7.2%) |
| Validation, the 37 series Sunday scores | 273 | | | +14.7% (+6.3% to +25.2%) |

- **The offer lists are exact.** The median over a day's offers reproduces the pack's official
  value on every day of all 592 posted rental series.
- **The port is exact.** It reproduces the sister project's recorded 6 Nov engine forecast for
  the 37 headline series to the last digit (111 of 111 values).
- **The gain is clear on the training window and not established on validation.** On the 37
  series Sunday scores it is larger. Those 37 are the starter kit's headline set, chosen with
  data to 6 Nov, so that row has hindsight.
- **This does not independently confirm the sister project.** Its 8% was measured on those series
  over 21 Aug to 5 Nov, which overlaps our validation and locked-test dates.
- **Timing does not seem to be the source.** In five copies of the validation data with the order
  of every list's price changes shuffled, the engine beat drift by 1.3% to 7.0%. The control is
  rough: a copy keeps about four in five of the real price changes, and the real gain it is
  compared with is itself not established.
- **No per-class rule is used.** `neocloud_spot` is −7% on the training window and +23% on
  validation; `marketplace_ask` is +10% and then −9%.
- **Its ranges are close to target on validation:** 84% of truths inside [q10, q90], against a
  target of 80%.

## 5. What the system learned across iterations
*Fills from the research notebook and the archive.* The seed agenda, tested on data up to 30 Sep:

| Hypothesis | Result | Verdict |
|---|---|---|
| Neocloud rate cards usually do not move within 7 days | 87% of 7-day windows unchanged (p = 0.001) | kept |
| Large recent moves are followed by large moves | corr(\|past 14-day move\|, \|next 7-day move\|) = +0.09 (p = 0.002) | kept |
| Spot prices revert towards their 30-day median | corr = −0.24 (p = 0.012) | kept |
| Moves continue (momentum, 7 days) | corr = −0.30: the opposite sign; short-term moves partly reverse | rejected |
| A falling provider count precedes price rises | corr = −0.16 (p = 0.11) | rejected |
| Collectors of the same GPU follow each other with a lag | corr = −0.08 | rejected |

**What moves a median** (data to 30 Sep, 167 series; `gpr.offers.median_moves`). Each day-to-day
move is split, in order, into three causes:

| | Offers joining or leaving | Frequent repricers | Other repricing |
|---|---|---|---|
| All classes | 31% | 45% | 24% |
| Neocloud rate cards (50 series) | 57% | 3% | 40% |
| Hyperscaler lists (17) | 93% | 0% | 7% |
| Marketplace asks (23) | 28% | 68% | 4% |

- **A move in a posted median is often not a provider changing its price.** For rate cards, more
  than half of the movement is the sample changing.
- **This is what the offer list on the board is for:** an agent that sees who sits in the middle,
  and who just joined or left, can tell the two apart.

**Carried over from the sister project as starting points, untested here.** They were chosen
there after its weeks from 16 Oct to 5 Nov had been looked at, and those weeks include our
locked-test origins. So a locked-test score for `configs/team_list.yaml` would not be blind.
- Ranges stretched by 1.25 at 3 days and 1.65 at 7 days. That was the larger of the two changes
  its coach kept. The Improver may now set `rule.stretch_1`, `stretch_3` and `stretch_7`.
- No debate round. There, letting agents read each other's answers did not help (1.3% worse,
  −5% to +3%).
- Five list-reading jobs for the Forecasters. There, five GPT agents with these jobs, averaged,
  cut error by 15% to 19% against drift.

## 6. The 2–3 year view (argued, not scored)
Full model and tables: [results/long_run.md](results/long_run.md), built by
`python scripts/long_run.py` from the pack only.

- **Today (GIX index, 3 Nov 2026):**
  - H100 $3.33/GPU-h, A100 $1.98, H200 $4.28, B200 $7.32.
  - Per unit of measured inference work (MLPerf: H100 = 1, B200 = 3.21, A100 = 0.41), a
    B200-hour costs the equivalent of $2.28 per H100-hour. So H100 is priced 46% above B200
    parity, and A100 114% above. Older chips are expensive per unit of work and under
    downward pressure.
- **What the market itself implies:**
  - GetDeploying's 36-month reserved H100 price is 0.77× on-demand (A100 0.76×). Read as the
    average rent a lessor expects over the term, that is about **−17%/yr** for both chips.
  - B200's 36-month price is only 0.91× on-demand, and its 12-month price is above on-demand.
    That points to scarcity for the newest generation.
- **Scenarios:**
  - Rent changes: low −35%/yr (AWS H100 spot after launch), base −17%/yr (market-implied),
    high −4%/yr (A100 AWS spot over four years).
  - Floor: power and operations, $0.13–0.34/GPU-h for H100.
  - Resale: present value of the remaining net earnings to end of life.
- **H100 in Nov 2029:** rent $0.92 / **$1.94** / $2.95 (low / base / high). Modelled resale
  $1.3k / **$16.7k** / $39.6k: about 50% of its $33.6k launch price in the base case.
- **A100 in Nov 2029:** rent $0.54 / **$1.12** / $1.75. Modelled resale $0.2k / **$3.4k** /
  $7.7k: 22% of the $15k launch price (40 GB SXM4) in the base case.
- **For a lessor or lender:**
  - The value hinges on the rent path, not on the hardware. The low and high cases differ by
    a factor of about 30 for H100 resale.
  - A collateral haircut should be set against the low case.
  - The market-implied decline (the reserved-to-on-demand ratio) is observable every week: it
    is the cheapest leading indicator in the pack.
  - The high case's H100 resale (118% of launch) is a sign that holding rents near today's
    level for three years is optimistic, not a forecast.
- **Not in the model:**
  - Transacted prices: none are public, and all pack prices are posted.
  - Financing costs.
  - A demand shock.

Resale values are modelled, not quoted. Compare them with the publishers listed in the pack's
`resale_pointers.csv`, which are cited but whose values are not stored, per DATA.md section 9.

## 7. Reproduce
`bash reproduce.sh`: it verifies the pack, runs the tests, runs the locked test and the
ablations, writes `submission/forecast_2026-11-06.csv`, checks the list-replay engine
(`results/engine.md`, no LLM), and builds `results/report.md`.
`MODE=base` needs no checkpoint; `MODE=full` reruns the whole loop. Sampling seeds are fixed.

## 8. Limits
- Every price in the pack is posted, not transacted.
- The checkpoint lives in our Tinker account. A rerun from scratch (`MODE=full`) needs Tinker
  and hours of compute.
- The newer Tinker base models may have seen some 2026 price pages in pretraining. The research
  tests and the baselines don't depend on that, but the LLM forecasts could.

Sources and licences: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md), copied unchanged from the pack.
