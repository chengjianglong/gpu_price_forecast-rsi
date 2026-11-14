# human-centric-world-model

How long can a generated world hold together while an agent plays in it? This repo takes open
interactive world models, **ABot-World-0** and **Matrix-Game 3.0**, and works on making them hold
together longer through action calibration, memory, state tracking and re-anchoring, plus a small
fine-tune. Nothing is trained from scratch. Progress is measured on
[PlayWorld](https://github.com/kxding/PlayWorld).

**Target.** Genie 3 (closed) leads PlayWorld at **2.12** overall; Matrix-Game 3.0 scores **1.14**.
The aim is an open model above 2.12. See [docs/PLAN.md](docs/PLAN.md) for the plan.

| Model | GC | IF | IE | OE | Overall |
|---|---|---|---|---|---|
| Genie 3 (closed, record) | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |
| LingBot-World2 (open) | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| Matrix-Game-3.0 (open) | 1.30 | 1.25 | 1.00 | 1.00 | 1.14 |
| ABot-World-0 (ours, baseline) | – | – | – | – | not run yet |

GC = geometry consistency, IF = interaction fidelity, IE = insight evolution, OE = out-of-sight evolution.
Each is scored 1–5 by PlayWorld's Gemini 3.1 Pro judge; Overall is their mean.

## Status

This is the scaffold: the closed-loop harness, adapters for both models, an action-calibration
probe and a scoring wrapper. It has been tested end to end on CPU with a mock world only.
**Neither model adapter has been run on a GPU yet.** They follow each repo's own inference code at
the pinned commits, but expect fixes on first contact.

## Layout

```
hcwm/
  actions.py        PlayWorld action space, reference plans, insight (wait-only) detection
  tasks.py          load PlayWorld cases (gc / if / insight / outsight splits)
  agent.py          agent player: Claude Haiku 4.5 (as in PlayWorld) or the scripted reference plan
  runner.py         closed-loop rollout, PlayWorld output layout, Hook extension point
  calibrate.py      measure yaw/pitch per action from generated frames; suggest gains
  backends/
    base.py         WorldModel interface: reset(image, prompt) / step(action) -> frames
    abot.py         ABot-World-0, block by block (W A S D + I J K L key states)
    matrixgame3.py  Matrix-Game 3.0, its generate loop in a thread with actions injected
    mock.py         CPU stand-in with known camera motion, for tests
configs/            one YAML per backend
scripts/
  setup_third_party.sh   clone PlayWorld, ABot-World, Matrix-Game at pinned commits
  run_playworld.py       run cases in closed loop
  calibrate_actions.py   action -> camera-motion probe
  score_playworld.py     PlayWorld judge, N passes, PlayWorld's own aggregation
tests/              CPU tests on the mock world
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev,agent]"
.venv/bin/python -m pytest -q
./scripts/setup_third_party.sh
```

Plumbing check on CPU with PlayWorld's bundled example (no GPU, no API key):

```bash
.venv/bin/python scripts/run_playworld.py --config configs/mock.yaml --agent scripted \
  --data-root third_party/PlayWorld/example --dataset third_party/PlayWorld/example/GC/001.json --run mock_check
```

### On a GPU machine

1. Install each model's environment from its README (PyTorch, FlashAttention, ...) and download weights
   (commands printed by `setup_third_party.sh`).
2. Put the full PlayWorld dataset under `data/playworld/{gc,if,insight,outsight}/data.json`.
3. Set `ANTHROPIC_API_KEY` (agent) and `GEMINI_API_KEY` (judge); see `.env.example`.

```bash
# baseline rollouts
python scripts/run_playworld.py --config configs/abot.yaml --data-root data/playworld --run abot_baseline
python scripts/run_playworld.py --config configs/matrixgame3.yaml --data-root data/playworld --run mg3_baseline

# does a turn key actually turn the camera?
python scripts/calibrate_actions.py --config configs/matrixgame3.yaml --images data/playworld/gc/*.jpg --target-deg 30

# judge, 3 passes, PlayWorld's aggregation (Fail = 1)
python scripts/score_playworld.py --run abot_baseline --passes 3
```

## Protocol notes

- The agent follows PlayWorld's local-model harness (its SANA-WM adapter): one decision per step,
  at most 40 steps. The agent sees the initial frame, samples from earlier steps and the latest step, and
  the reference plan's suggestion, then answers with one action word. Insight cases skip the agent and
  hold WAIT for 60 s.
- The world model is prompted with the case's scene caption by default (`prompt_source` in the config);
  the objective goes to the agent only.
- Matrix-Game frames are captured before the repo draws its key/mouse overlay, so neither the agent nor
  the judge sees it.
- Keep the agent model fixed (Claude Haiku 4.5) when comparing against the leaderboard.

## Next

Week-one goal: baselines for both models on the full 171 cases, then pick the starting model
(rule in the plan). After that, in order: action calibration, the long-session drift test,
3D memory with re-anchoring, the out-of-sight object tracker, the still-scene guard, and a LoRA
fine-tune. Each plugs in as a `runner.Hook` or a backend option and is measured one at a time.

Built on [human-centric-world-simulation](https://github.com/chengjianglong/human-centric-world-simulation),
whose Blender world will serve as a ground-truth test bed for drift and object identity.
