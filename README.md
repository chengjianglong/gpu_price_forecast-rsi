# gpu-pricing-rsi

A team of role-based agents that forecasts GPU rental prices (per GPU-hour) and improves itself
round by round against a backtest. Built for Track 2 (Recursive Self-Improvement) of the
iterate.inc AI research hackathon, running on Tinker.

Work in progress: the code is being scaffolded.

## Design
- **Target:** daily median $/GPU-hour for H100, H200, B200 and A100 (on-demand and spot),
  1, 7 and 30 days ahead, as 10th/50th/90th percentile forecasts.
- **Agents:** Data Scout, Quant Analyst, Market Analyst, three Forecasters with different priors,
  Critic, Aggregator, Scorer (code only) and Improver.
- **Loop:** backtest with time-sliced data → diagnose errors → Improver proposes candidate changes →
  keep a change only if it beats the current version on a validation window → periodically
  fine-tune the forecaster on Tinker with reward = −pinball loss → repeat. A final test window
  stays untouched until the end.
