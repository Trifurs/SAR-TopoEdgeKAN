# Publication discussion experiments

These optional configurations cover input fairness, edge-KAN specificity, hydrostatic-prior and SAR contributions, and inference efficiency. Implementations, configurations, commands and tests live in the corresponding `discuss/` directories. They are opt-in and are not added to the default seven-model catalog.

All trainable configurations inherit the existing dataset contract, split, train-only normalization, augmentation, supervision domain, task-adaptive loss, AdamW settings, learning-rate schedule, batch size, epoch/early-stop rules and checkpoint selection criterion. They use the same production trainer and the same paired seed schedule. Checkpoints are selected on validation data; comparisons use the test split. No configuration changes the dataset split or adds a loss term.

## Configuration groups

| Group | Configurations in `configs/discuss/` | Purpose |
|---|---|---|
| Shared reference | `full.xml` | SAR-TopoEdgeKAN, trained through the discussion protocol. |
| Input fairness | Five `*_matched.xml` comparators; shared reference | Expose every observation used by the full model to the comparator architectures. |
| Edge mechanism | `no_graph.xml`, `fixed_edges.xml`, `linear_edges.xml`, `mlp_edges.xml`; shared reference | Distinguish graph aggregation, fixed topographic affinity, linear edges, nonlinear MLP edges and KAN edges. |
| Prior and SAR | `prior_only.xml`, `no_prior.xml`, `terrain_only.xml`, `unet_shared_head.xml`, `unet_prior.xml`; matched U-Net and shared reference | Separate deterministic geometry, prior residual prediction, SAR observations and backbone effects. |
| Efficiency | `efficiency.xml` | Common GPU measurement protocol for all unique discussion models. |

There are **14 unique trainable configurations and one deterministic control**. References shared by several groups are deduplicated when groups are run together. Five paired seeds across all trainable configurations require 70 training jobs; the prior-only model is evaluated once.

## Input fairness

The five matched comparators receive 19 channels assembled from the same observation whitelist as the full model:

| Observations | Channels |
|---|---:|
| Normalized pre/event/change S1 | 7 |
| Normalized DSM and slope | 2 |
| Raw DSM and slope | 2 |
| S1 validity, event support, DEM validity | 3 |
| Per-branch valid fractions | 4 |
| Supplied known flood range | 1 |

Raw terrain is scaled by fixed unit conversions inside the comparator adapter (DSM divided by 1000, slope divided by 90). It is not clipped and no new normalization statistics are estimated. The original metric-valued DSM is used directly by the shared hydrostatic prior. Labels, event IDs, QA values not consumed by the full model and other label masks are excluded. Architectures may process the shared observations differently.

Every input-facing convolution is expanded, including residual skip projections. Weights for the original nine normalized observation channels and range channel are retained at initialization; new channel weights start at zero and are trainable. All matched comparators and the full model use the same 19-channel observation whitelist; no dataset-contract exception is permitted by the protocol validator.

## Edge-function controls

The graph topology, six physical descriptors, graph stride, latent compatibility, observation gates, message mixer and bounded graph residual are shared. KAN and MLP edge corrections are zero-output initialized. Common non-edge parameters and the subsequent RNG state match for the same seed.

- `mlp_edges.xml`: replace each 6-to-1 KAN edge function with a 6-7-1 SiLU MLP. The two heads contain 114 MLP parameters versus 110 KAN parameters, a difference of four in the whole model.
- `linear_edges.xml`: replace the KAN edge function with a learned linear 6-to-1 map.
- `fixed_edges.xml`: remove the learned edge correction and freeze the initial physical-affinity coefficients as buffers. Observation gates, latent compatibility, message mixers and residual strength still learn; this is a fixed **topographic affinity**, not a completely fixed graph operator.
- `no_graph.xml`: physically remove the entire graph module, including inactive parameters. This produces the same prediction path as the production graph-removal ablation while reporting its actual reduced resident size.

Fixed affinity removes learning of the physical-affinity coefficients as well as the edge correction. The clean near-parameter-matched KAN-specific comparison is therefore **KAN versus MLP**, not KAN versus the fixed variant alone. All controls use only the same [shared output-depth objective](../COMPARISON_PROTOCOL.md#shared-depth-objective).

## Prior and sensor controls

- `prior_only.xml` uses the exact production connected-component boundary-quantile prior. It has zero parameters and no optimizer, checkpoint or repeated-seed fitting stage.
- `no_prior.xml` uses the production conditional-positive head without the hydrostatic residual reference.
- `terrain_only.xml` physically removes the SAR encoder, SAR alignment and change-injection projection. Terrain features provide the decoder backbone and DEM validity supplies graph support. No SAR values or S1 validity tensors enter its prediction. The evaluator keeps the same canonical valid-pixel domain for all methods.
- `unet_shared_head.xml` and `unet_prior.xml` use the same U-Net backbone, matched observations and the production conditional-depth head design. The latter enables the exact production hydrostatic prior and bounded multiplicative log-depth residual. Include the shared direct head to separate prior effects from changing the native one-layer U-Net output head.

The supplied range is still derived from target support. These configurations evaluate known-range depth reconstruction; they do not establish robustness to predicted extent errors or independence between dataset event chains.

## Commands

Run from the repository root in an environment with CUDA-enabled PyTorch. The controller applies the same isolated-worker MKL policy as the production seed runner.

```bash
export MKL_THREADING_LAYER=GNU

# Inspect and verify protocols/parameter counts without training.
python tools/discuss/run_experiments.py check --build

# Run all 14 trainable configurations with five paired seeds.
python tools/discuss/run_experiments.py train --groups all --seeds 5

# Or select one group; references are deduplicated within this command.
python tools/discuss/run_experiments.py train --groups input --seeds 5
python tools/discuss/run_experiments.py train --groups graph --seeds 5
python tools/discuss/run_experiments.py train --groups prior --seeds 5

# Evaluate the deterministic prior on the full test split once.
python tools/discuss/run_experiments.py prior

# Train a single configuration through the existing entry point.
python train.py configs/discuss/mlp_edges.xml
```

Discussion seed campaigns are written below `runs/sar_topo_edge_kan/discuss/random_seed_validation/`. Resume a particular session with `train --seeds 5 --resume <session-directory>`; select the same groups as the original session. Existing code/config/data fingerprint checks reject incompatible resumes. The deterministic control must use the `prior` command, not `train.py` or the general seed runner.

Do not mix ten-channel comparator scores with the matched-observation discussion table. Read `metrics_by_seed.csv` and `metrics_summary.csv` from a common discussion campaign.

## Efficiency protocol

After completing the desired discussion training runs:

```bash
python tools/discuss/profile_efficiency.py \
  --config configs/discuss/efficiency.xml \
  --runs runs/sar_topo_edge_kan/discuss
```

The configuration uses the same 256-pixel test samples, GPU, BF16 precision, batch sizes 1 and 12, ten warm-up forwards and three repetitions of 100 timed forwards. `checkpoint_seed_index=1` selects the first paired seed for every model; within that seed, the newest completed checkpoint for each unique experiment ID is selected. The exact seed, checkpoint and sample IDs are recorded. Required model operations, including the hydrostatic prior, are timed with CUDA synchronization. Data IO and evaluation-only physical diagnostics are excluded. Parameter size, trainable size, executed-module parameters, peak allocated/incremental GPU memory, median/p95 batch latency, amortized sample latency and throughput are reported.

Stage timings are collected in a separate synchronized pass for the SAR encoder, terrain pyramid, graph, decoder, prior, head and comparator backbone where applicable. Hooks introduce overhead and nested stages must not be added together. Stage timings explain cost; the independent unhooked measurements provide total latency.

`supported_flops_per_sample` counts only operations supported by `torch.profiler`; it is **not complete model FLOPs**. The output lists counted operators and operators without FLOP counts. Sorting/quantiles, pooling, elementwise operations and control flow can dominate the prior despite low counted FLOPs. Do not label this field as complete FLOPs/MACs in a paper.

For a short architecture-only diagnostic before training, explicitly use `--allow-untrained` and smaller iteration counts. Such outputs are marked `untrained_architecture` and do not provide trained-model accuracy evidence. An example is:

```bash
python tools/discuss/profile_efficiency.py \
  --models full no_graph mlp_edges prior_only \
  --allow-untrained --batch-sizes 1 --warmup 1 --iterations 2 --repetitions 1
```

The JSON report retains per-forward timing samples and nested diagnostic details; `efficiency.csv` contains scalar fields. Output directories are created under `runs/sar_topo_edge_kan/discuss/efficiency/` unless explicitly selected.

## Verification

```bash
python -m pytest -q tests/discuss
```

Tests cover shared protocol enforcement, observation inclusion, graph/MLP initialization, physical removal of inactive branches, prior identity, real-sample forward passes, the shared loss/optimizer backward step and edge-function gradients. Available CUDA is used for the model/gradient checks. Passing these tests verifies implementation contracts; module effectiveness remains an empirical result of the full paired training and test experiments.
