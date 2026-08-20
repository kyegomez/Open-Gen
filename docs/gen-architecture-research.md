# GEN Architecture Research

Research synthesis covering Generalist AI's GEN-0 through GEN-1.5, updated
August 20, 2026.

> GEN's public behavior is unusually well documented, but its implementation is
> not. This report separates company disclosures from technically plausible
> reconstruction. Architecture confidence is medium.

## Bottom line

The strongest defensible classification is an attention-based, native multimodal
sensorimotor sequence model. Calling it a Transformer is a high-confidence
inference from token streams, context prompting, attention, and paged attention;
Generalist has not explicitly disclosed the backbone.

The minimum architecture consistent with the evidence is:

```text
Video · language · proprioception · other sensors
                         │
                         ▼
              Sensor-processing front end
                         │
                         ▼
       Continuous-time event and token alignment
                         │
                         ▼
          Harmonic Reasoning temporal backbone
                         │
                         ▼
       Embodiment-conditioned actuation pathway
                         │
                         ▼
             100 Hz trajectory samples
```

“Produces 100 Hz action trajectories” does not establish that the complete
10B+ network performs one forward pass every 10 milliseconds. A prediction may
contain multiple trajectory samples that execute while a slower model computes
the next prediction.

## Evidence ledger

| Question | Best-supported answer | Status | Source |
|---|---|---|---|
| Model family | Large multimodal embodied foundation model; approximately 99% of GEN-1 is trained from scratch | Direct | GEN-1 / Beyond VLAs |
| Inputs | Video, language, proprioception, and other sensor streams | Direct | GEN-1.5 |
| Memory | 30-second context containing physical prompts and rolling observations | Direct | GEN-1.5 |
| Core | Asynchronous, continuous-time sensing and acting token streams called Harmonic Reasoning | Direct | GEN-0 |
| Subsystems | Sensor processing → harmonic reasoning → actuation | Direct | Thousand Hands |
| Output | 100 Hz action trajectories; model invocation rate is undisclosed | Direct with caveat | GEN-1.5 |
| Backbone clue | Custom kernels and new forms of paged attention for real-time inference | Direct | GEN-1 |
| Scale | GEN-0 tested at 1B, 6B, 7B, and later 10B+ parameters | Direct | GEN-0 |
| Training data | 270k hours for GEN-0; 500k+ for GEN-1, primarily human wearable-device interaction | Direct | GEN-0 / GEN-1 |
| Pretraining signal | Random continuous sensorimotor spans; evaluation uses next-action prediction error | Direct, objective details withheld | GEN-0 / GEN-1.5 |

## What remains undisclosed

| Area | Missing information |
|---|---|
| Backbone | Transformer variant, depth, width, and MoE/dense topology |
| Vision | Encoder, resolution, frame rate, and temporal token compression |
| Action | Autoregressive vs. diffusion/flow, chunk horizon, and dimensions |
| Timing | Model refresh rate, timestamp encoding, and stream synchronization |
| Embodiment | Robot schema, normalization, masks, and adapter design |
| Losses | Exact pretraining objectives and optimization recipe |
| Language | Tokenizer, corpus, pretraining fraction, and functional role |
| Deployment | Quantization, hardware, latency, and KV-cache policy |

## Reconstructed architecture

### Sensor front end

Modality-specific encoders likely convert video, language, robot state, and
historical actions into a common embedding space. The named architectural
decomposition—sensor processing, Harmonic Reasoning, and actuation—supports a
distinct sensor front end rather than raw modality values entering one
undifferentiated network.

### Event and token alignment

The phrase “asynchronous, continuous-time streams” suggests that observations
and actions need not share a single fixed token clock. Plausible implementations
include:

- Explicit timestamps attached to every event.
- Continuous or relative-time positional embeddings.
- Modality identifiers and camera/sensor identifiers.
- Attention masks based on physical time rather than sequence index.

This does not imply a neural ODE. Timestamped event sequences are sufficient.

### Harmonic Reasoning

A causal or streaming Transformer-family model is the most likely core because
GEN uses:

- Sensing and acting tokens.
- A 30-second context.
- In-context sensorimotor examples.
- Attention and paged attention.
- Randomly sampled continuous pretraining spans.

Generalist contrasts Harmonic Reasoning with Helix-style explicit fast/slow
modules and Real-Time Chunking-style inference guidance. This suggests that
overlapping sensing, reasoning, and action is learned more tightly end to end,
but the exact mechanism remains private.

### Actuation

The action path must represent a continuous, potentially multimodal policy.
Generalist reports policy samples, reverse-KL estimates, next-action prediction
error, and 100 Hz trajectories. Diffusion, flow matching, autoregression,
energy-based prediction, and hybrid objectives all remain possible.

### Evolution

| Generation | Public architectural advance | Data / scale | Adaptation |
|---|---|---|---|
| GEN-0 · Nov 2025 | Harmonic Reasoning, cross-embodiment support, scaling laws | 270k hours; 1B–10B+ | Thousands of post-training steps |
| GEN-1 · Apr 2026 | Evolved inference, paged attention, RL, policy steering | 500k+ hours; size undisclosed | Approximately one hour of robot data |
| GEN-1.5 · Aug 2026 | 30-second physical-prompt context | Eight+ months of continual pretraining | Zero steps or 1–10 gradient steps |

## Technical lineage

Prior work by Generalist's team makes particular mechanisms plausible without
proving GEN uses their implementations.

| Prior work | Team overlap | Relevant mechanism | Relationship to GEN |
|---|---|---|---|
| PaLM-E · 2023 | Florence, Zeng et al. | Projects image, state, and language into one decoder-only Transformer sequence | Strong conceptual ancestry for sensing tokens |
| RT-2 · 2023 | Florence et al. | Makes discretized robot actions autoregressive VLM tokens | Strong acting-token ancestry; GEN rejects the VLM-fine-tuning recipe |
| Interactive Language · 2022 | Florence et al. | A continuously running policy accepts changing language while acting | Strong real-time conditioning precedent |
| ALOHA Unleashed · 2024 | Florence et al. | Transformer diffusion predicts one-second, 50 Hz bimanual chunks | Strong trajectory precedent; no proof GEN uses diffusion |
| General Pattern Machines · 2023 | Florence, Zeng et al. | LLMs continue numeric state/action trajectories in context | Intellectual precursor to physical prompting |
| XIRL · 2021 | Zeng, Florence et al. | Embodiment-invariant task progress from human and robot videos | Strong human-to-robot transfer lineage |
| Implicit Kinematic Policies · 2022 | Florence, Zeng et al. | Joint and Cartesian action representations | Possible cross-embodiment influence |
| Video Language Planning · 2023 | Florence, Zeng et al. | Generated video futures as a learned dynamics model | World-model lineage, but unlike GEN it uses explicit search |
| VIRDO++ · 2022 | Zeng, Florence et al. | Visuo-tactile state and dynamics prediction | Physical-state lineage |
| Boston Dynamics systems | Barry et al. | High-rate control and coordinated arm-body manipulation | Likely systems influence, not model-architecture evidence |

There is no evidence that GEN specifically uses diffusion, flow matching, ACT
ensembling, FAST/DCT tokenization, a neural ODE, explicit world-model rollouts,
or any named prior patent.

## Comparison with public robot-policy architectures

| System | Backbone | Timing architecture | Action decoder | Control claim | Memory |
|---|---|---|---|---|---|
| GEN | Native sensorimotor model | Unified asynchronous streams | Undisclosed | 100 Hz trajectories | 30 seconds |
| π0 | PaliGemma VLM + action expert | Shared attention, separate expert weights | Conditional flow-matching chunks | Up to 50 Hz | Short observations |
| Helix | Dual-system VLA | 7–9 Hz semantic + 200 Hz visuomotor | Fast continuous-control head | 200 Hz low-level | Split timescales |
| RTC / π0.5 | Flow-policy runtime method | Overlapping predicted chunks | Flow-matching chunks | Real-time execution | Chunk history |
| RT-2 | PaLI-X / PaLM-E VLM | Image + language → action tokens | Autoregressive 256-bin actions | 1–5 Hz model | Current image |
| OpenVLA | DINOv2 + SigLIP + Llama 2 7B | Vision-language late fusion | Autoregressive 256-bin actions | 5–15 Hz tests | Single image |
| Octo | 27M / 93M blockwise Transformer | Modular observation/task tokenizers | Diffusion head, four-action chunks | Embodiment-dependent | Two observations |
| GR00T N1 | VLM + separate DiT | Explicit slow/fast dual system | Flow matching, 16-step chunks | VLM approximately 10 Hz | Current observation |
| RDT-1B | 1.2B Diffusion Transformer | Unified state/action schema | DDPM, 64-action chunks | 381 actions/s generated | Two observations |
| Gemini Robotics | Cloud VLA + local decoder | Cloud reasoning, local execution | Action chunks; objective undisclosed | Effective 50 Hz | Current scene |
| ACT | Approximately 80M CVAE Transformer | Current observation → parallel queries | Direct regression, 100-step chunks | 50 Hz playback | Current observation |

## Independent validation audit

GEN's systems and demonstrations are credible evidence that the models exist,
but every architecture, scaling, and benchmark result remains company-reported.
Independent articles contextualize Generalist's numbers; they do not reproduce
them.

| Claim area | Assessment | Confidence |
|---|---|---|
| Public artifacts | No weights, code, dataset, evaluation harness, formal model card, or GEN paper | High |
| Architecture | No layer topology, action decoder, tokenization, or inference cadence is auditable | High |
| 7B phase transition | Internal checkpoints and tasks; raw curves and fit details unavailable | High |
| GEN-1 99% | Selected internal tasks; comparator is primarily Generalist's GEN-0 | High |
| GEN-1.5 59% / 83% | Arithmetic is consistent; trial counts, seeds, confidence intervals, and selection protocol are absent | High |
| Reported ± values | Appear to be standard deviation across task rates, not uncertainty in the mean | Medium-high |
| 100 Hz | Trajectory sample rate disclosed; full-model inference frequency undisclosed | High |
| “No robot data” | Refers to base pretraining; task adaptation still uses robot data | Medium-high |

### Documented inconsistencies

- The GEN-1 page refers to a “March 2025” GTC demo using advances made since
  November 2025. This impossible chronology almost certainly means March 2026.
- Secondary articles sometimes call GEN-0 “GEN-θ.” Generalist consistently uses
  GEN-0; the alternate name appears to be a rendering error.
- GEN-1 and GEN-1.5 parameter counts are undisclosed. Calling either model 10B+
  extrapolates from GEN-0.

### Evidence that would improve confidence

1. Frozen checkpoints and inference code with timing traces.
2. Public task definitions, trial counts, seeds, and complete rollout logs.
3. Dataset documentation and contamination checks for “unseen” tasks.
4. Standardized comparisons against π0.5, GR00T, OpenVLA, and Octo.
5. External evaluation across new sites, objects, operators, and embodiments.

## Open implementation blueprint

This is a practical open architecture targeting GEN's disclosed behavior, not a
claim about Generalist's private implementation.

| Module | Recommended implementation | Reason |
|---|---|---|
| Vision | MoonViT or SigLIP/DINOv2 + temporal token resampler | Efficient multi-camera video representation |
| State/action | Masked MLP projections + embodiment/schema tokens | Variable DoF and tool support |
| Time | Continuous timestamp and duration embeddings | Asynchronous sensor/action events |
| Core | Blockwise-causal Transformer over timestamped events | Long multimodal context |
| Memory | Paged KV cache + hierarchical 30-second buffer | Dense recent video, pooled older video, exact actions/states |
| Action | Flow-matching expert producing 0.5–1 second chunks | Multimodal, precise trajectories |
| Runtime | Asynchronous chunk refresh with overlap and blending | Decouple 100 Hz control from model latency |
| Training | Next-action/chunk prediction + optional latent sensor prediction | Scalable sensorimotor sequence learning |
| Adaptation | Full or LoRA SFT followed by offline-to-online RL | Few-step task and embodiment adaptation |

Recommended implementation order:

1. Validate video/state-to-action learning on one embodiment.
2. Replace fixed frame stacks with timestamped event tokens.
3. Train long contexts containing demonstrations and rollouts.
4. Mix embodiments, tools, camera arrangements, and action dimensions.

## Sources

### Primary

- [GEN-0](https://generalistai.com/blog/gen-0)
- [GEN-1](https://generalistai.com/blog/gen-1)
- [GEN-1.5](https://generalistai.com/blog/gen-1.5)
- [Going Beyond World Models & VLAs](https://generalistai.com/blog/beyond-world-models)
- [Towards Machines with a Thousand Hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)

### Technical lineage

- [PaLM-E](https://arxiv.org/abs/2303.03378)
- [RT-2](https://arxiv.org/abs/2307.15818)
- [Interactive Language](https://arxiv.org/abs/2210.06407)
- [ALOHA Unleashed](https://arxiv.org/abs/2410.13126)
- [Large Language Models as General Pattern Machines](https://arxiv.org/abs/2307.04721)
- [XIRL](https://arxiv.org/abs/2106.03911)
- [Video Language Planning](https://arxiv.org/abs/2310.10625)

### Independent audit

- [DEPLOY registry](https://registry.deploy.report/brains/gen-0)
