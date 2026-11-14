# Track 4 World Model Plan

Nov 14, 2026. Living version: the Claude Doc "Track 4 World Model Plan"; this is a snapshot.

Goal: take an open world model (ABot-World-0 or Matrix-Game 3.0) above Genie 3's PlayWorld overall score of 2.12 without training from scratch, using action calibration, 3D memory with re-anchoring, an out-of-sight object tracker, and a small LoRA fine-tune.

## Target

We need an overall of at least 2.13, the mean of four dimensions each scored 1–5. One mix that gets there: GC 2.5, IF 2.2, IE 1.8, OE 2.0 (sum 8.5).

| Model | GC | IF | IE | OE | Overall |
| --- | --- | --- | --- | --- | --- |
| Genie 3 (closed, record) | 2.74 | 2.40 | 1.51 | 1.81 | 2.12 |
| LingBot-World2 (open) | 2.04 | 2.13 | 1.95 | 1.16 | 1.82 |
| LingBot-World (open) | 2.11 | 2.23 | 1.33 | 1.43 | 1.78 |
| Matrix-Game-3.0 (open) | 1.30 | 1.25 | 1.00 | 1.00 | 1.14 |
| Our target | 2.50 | 2.20 | 1.80 | 2.00 | 2.13 |

How PlayWorld scores:

- **Dimensions.** GC = geometry consistency on camera moves and revisits. IF = interaction fidelity. IE = insight evolution: the camera holds still for 60 s and the process in view must progress. OE = out-of-sight evolution: an object leaves view and must come back as itself, changed plausibly.
- **Rubric.** Gemini 3.1 Pro answers weighted yes/no questions per case; score = 1 + 4 × (weighted yes fraction).
- **Gate.** A rollout that fails trajectory or subject validation scores 1 on that dimension.
- **Player.** Claude Haiku 4.5 plays 171 cases with W/A/S/D, arrow keys and WAIT, at most 40 steps and 10–60 s per case. It follows a human-written preset sequence and edits about 12–15% of actions.

Why the target is reachable: Matrix-Game 3.0 sits at the floor (IE and OE exactly 1.00) mostly by failing the gate. Its geometry trajectory pass rate is 37.5% and its rotation pass rate 18.1%. Genie 3 is weakest on IE (1.51) and OE (1.81), so those two dimensions are where an open model can overtake it.

## Starting model

Measure baselines for both models in week 1, then commit to one. Recommended default: ABot-World-0, with Matrix-Game 3.0 kept as a second result.

| | ABot-World-0 5B | Matrix-Game 3.0 5B |
| --- | --- | --- |
| Licence | Apache-2.0 | Apache-2.0 |
| PlayWorld overall | Not scored yet | 1.14 |
| Backbone | Wan2.2 DiT, action-conditioned | DiT with camera-aware memory, DMD-distilled |
| Speed (authors' numbers) | 720p, up to 16 FPS on one RTX 5090, ~19 GiB | 720p, up to 40 FPS |
| Long sessions | 24-hour continuous session claimed | Minute-long memory claimed |
| Text input | Yes (`set_prompts`, scene description) | Yes (prompt to `generate`) |
| Controls | 8 key states per ~1 s block: W A S D, I J K L | 6-way keyboard + per-frame [pitch, yaw] per 40-frame segment |
| Case for it | Higher likely ceiling; an unscored open model is new information | Largest possible gain over the starting model |

Decision rule: if ABot-World-0's baseline is at least 1.6, build on it. If not, build on Matrix-Game 3.0 and report gain over the starting model as the main claim. The brief names these two only as examples, so LingBot-World (open, 1.78) is a fallback if both baselines disappoint.

## Method

Five components, ordered by expected gain per unit of effort. Components 1–4 need no training; each is measured on its own before the next is added.

1. **Action calibration.** Targets the gate on GC, IF and OE.
   - Map W/A/S/D, arrow keys and WAIT to each model's native camera and movement inputs, with measured turn rate and step size.
   - Calibrate by estimating camera motion from generated frames and fitting the input scale so a turn step turns as far as the trajectory check expects (`scripts/calibrate_actions.py`).
   - Make WAIT advance time instead of freezing the frame.
   - Success check: translation and rotation pass rates, before and after.
2. **3D memory with re-anchoring.** Targets GC.
   - Per chunk, estimate depth and camera pose with an off-the-shelf model (VGGT or Depth Anything class) and fuse into a point-cloud memory.
   - Use pose integrated from calibrated actions as the prior; correct it with the visual estimate.
   - On a revisit, render the remembered view and feed it in: either as the model's memory frame (replacing Matrix-Game's own retrieval) or by partly noising it and denoising it, keeping the overlap region.
   - Anchor periodically to the first frame to stop slow drift in colour and texture.
3. **Out-of-sight object tracker.** Targets OE.
   - Segment and track objects; when one leaves view, log its identity crop, 3D position, state and time.
   - A language model predicts its state after the hidden interval (a car has moved on, a candle is shorter).
   - On return, project where it should be and condition generation with the reference crop, plus a text prompt.
4. **Keeping a still scene alive.** Targets IE.
   - Detect a still camera and stop collapse to a frozen frame: hold motion statistics across WAIT steps, add a text cue for the expected process, and re-inject recent frames.
5. **LoRA fine-tune.** Only after 1–4 show what errors remain.
   - Data: real walkthrough videos with loops and revisits, plus a small share of synthetic Blender clips with exact pose and object IDs.
   - Objectives: consistency on revisits, identity of objects that reappear, adherence to actions.
   - Keep the rank small and hold out PlayWorld-like scenes to check that the gains carry over.

## System at a glance

```
Agent player (Claude Haiku 4.5) --keys--> Action calibration --inputs--> World model (ABot-World-0 / Matrix-Game 3.0)
        ^                                                                       |
        |                         latest frame                                  v
        +------------------------------------------------------------- Generated frames --> video to the Gemini judge
                                                                                |
                        +-------------------------------+-----------------------+
                        v                               v                       v
                   3D memory                     Object tracker          Still-scene guard
          (depth + pose, revisit view,     (objects leaving view,     (still camera, keep the
           anchor to first frame)           state change, crop)        process moving)
                        |                               |                       |
                        +--------------> Conditioning back into the world model <+
```

The agent's keys pass through calibration into the world model. Each generated chunk updates the 3D memory, the object tracker and the still-scene guard. Those three feed conditioning back into the next chunk. In code, each is a `hcwm.runner.Hook` or a backend option.

## Reusing human-centric-world-simulation

The repo is a Blender world built from JSON plus a GPU navigation simulator, not a world model. Its value here is exact ground truth: we can measure where the generated world breaks instead of only asking a VLM judge.

| Part of the repo | Use in Track 4 |
| --- | --- |
| `blender/build_world.py`, `render_rollout.py`, `configs/world.json` | Render action-driven walkthroughs with exact camera pose, depth and object IDs: test data for the 3D memory, and a small share of LoRA data |
| `roomsim/traffic.py` (cars looping behind the house) | Ready-made out-of-sight cases with known true state |
| `roomsim/env.py`, `nav.py`, learned walking policies | Agents and people that move between rooms, giving revisit and reappearance cases |
| The trained policy in `runs/` | Optional extra: train it inside the world model, Dreamer 4 style, then test it in the real simulator |

What does not carry over: PlayWorld cases are photos of varied real scenes, so synthetic Blender data is mainly for diagnosis. The main LoRA data stays real.

## Evaluation and rigour

Every claim is backed by the full 171-case PlayWorld run, averaged over at least 3 judge passes (`scripts/score_playworld.py`). The paper's repeat scoring showed a per-model variance of about 0.011.

- **Ablations.** Score the baseline, then add components 1–5 one at a time, on all four dimensions and the gate pass rates.
- **Same cases as the record.** Report on the cases Genie 3 was scored on, so the comparison is like for like.
- **Judge check.** Re-score a sample with a second judge (e.g. Claude) and with a few human ratings, so the gain is not just tuned to Gemini's taste.
- **No gaming by stillness.** Report camera-motion statistics per run; the paper notes depth and subject scores stay high when a model barely moves.
- **Where the world breaks.** In the Blender test bed, run sessions of 1, 5, 10 and 30 minutes and plot against session length:
  - camera drift (generated pose vs true pose),
  - revisit match (PSNR/LPIPS against the true view),
  - share of objects that reappear as themselves.
- **Failure gallery.** For each dimension, the worst cases after all components, with a one-line cause each.

## Schedule

Assumes a start on Nov 16, 2026; the deadline is not yet known.

| Dates | Work |
| --- | --- |
| Nov 16 – Nov 22 | Harness + baselines for both models |
| **Nov 22** | **Gate: pick the starting model** (decision rule above) |
| Nov 23 – Nov 29 | Action calibration; long-session drift test |
| Nov 30 – Dec 13 | 3D memory + re-anchoring; object tracker |
| Dec 7 – Dec 13 | Still-scene guard |
| Dec 14 – Dec 20 | LoRA fine-tune; ablations + write-up |
| Dec 20 | Results frozen |

## Risks and open questions

| Risk | Effect | Fallback |
| --- | --- | --- |
| Local GPUs (2× RTX A2000) cannot run 5B video models at 720p | No baselines | Rent 1–2 H100/A100 (e.g. AWS) for baselines and 171-case runs |
| Memory injection breaks the distilled sampler | Drop in GC or speed | Use memory only as a retrieved frame; skip the partly-noised path |
| Gains tuned to one judge | Results do not hold up | Second judge plus human spot checks (Evaluation) |
| Blender data does not transfer to real scenes | LoRA gives no PlayWorld gain | Keep it to diagnosis; fine-tune on real video |

- [ ] Which GPU provider and budget?
- [ ] Anthropic key (agent player) and Gemini key (judge) in place?
- [x] Do ABot-World-0 and Matrix-Game 3.0 accept text alongside actions? Yes, both.
- [ ] Submission deadline?
- [ ] How many degrees per step does PlayWorld's trajectory check expect? Sets the calibration target.

## Sources

- [PlayWorld paper (arXiv 2608.13552)](https://arxiv.org/html/2608.13552v2)
- [PlayWorld code and leaderboard](https://github.com/kxding/PlayWorld)
- [Matrix-Game 3.0 paper (arXiv 2604.08995)](https://arxiv.org/html/2604.08995v2)
- [Matrix-Game 3.0 model card](https://huggingface.co/Skywork/Matrix-Game-3.0)
- [ABot-World-0 paper (arXiv 2607.19191)](https://arxiv.org/pdf/2607.19191)
- [ABot-World code](https://github.com/amap-cvlab/ABot-World)
- [human-centric-world-simulation](https://github.com/chengjianglong/human-centric-world-simulation)
