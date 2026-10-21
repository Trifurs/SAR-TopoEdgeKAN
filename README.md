# SAR-TopoEdgeKAN

SAR-TopoEdgeKAN is a Sentinel-1 and terrain flood-depth regressor with an eight-neighbour topographic edge KAN. Its trainable architecture, five learned comparators, three traditional comparators, and the edge-KAN removal ablation are provided here.

The experiment assumes a supplied flood range (`valid_depth_mask`). That range is derived from target support in this dataset. Report results as a **known-range/oracle benchmark**; the code does not predict inundation extent.

## Layout

- `models/`, `configs/sar_topo_edge_kan.xml`: main model.
- `configs/ablation/sar_topo_edge_kan_wo_tae_kan.xml`: edge-KAN removal.
- `compare/`, `configs/compare/`: five learned and three traditional comparators.
- `datasets/`, `assets/`: data loading and audited contract/statistics.
- `train.py`, `evaluate.py`, `test.py`: single-model training, evaluation, and common test suite.
- `validate_random_seeds.py`: fixed-split repeated-seed validation.

## Data and environment

FloodDepthNet raster files are not bundled. Before running an experiment, set the dataset root and manifest in `configs/base/datasets/flooddepthnet_s1_terrain.xml` to paths on your own machine, and prepare or verify the matching data contract and train-only statistics in `assets/flooddepthnet_s1_terrain/`. The audited fixed split contains 4,084 train, 584 validation, and 1,167 test samples; these counts apply only to the supplied manifest. Keep the manifest and normalization fingerprints consistent when relocating or changing the dataset.

From the repository root, activate a Python environment with CUDA-enabled PyTorch and install the dependencies:

```bash
export MKL_THREADING_LAYER=GNU
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Experiments

The default five-seed command runs the main model, all five learned comparators, and the formal edge-KAN ablation with the same fixed split and shared training protocol. It selects checkpoints on validation data and compares the selected checkpoints on the test split.

```bash
python validate_random_seeds.py 5
python validate_random_seeds.py --resume
```

For a focused run, pass `--models sar_topo_edge_kan wo_tae_kan`; this does not replace the complete comparison. Results are written under `runs/sar_topo_edge_kan/random_seed_validation/k5/<session>/`, including `metrics_by_seed.csv` and `metrics_summary.csv`.

To train or evaluate an individual learned model:

```bash
python train.py configs/sar_topo_edge_kan.xml
python evaluate.py configs/sar_topo_edge_kan.xml
python train.py configs/compare/deep_learning/dlsim_attention_unet.xml
```

To test all completed standalone learned runs and the three traditional methods on the same test split:

```bash
python test.py runs/sar_topo_edge_kan/train
```

See [model structure](docs/MODEL.md), [comparison protocol](docs/COMPARISON_PROTOCOL.md), and [comparison sources](docs/COMPARISON_SOURCES.md).

Optional publication controls for matched inputs, edge-KAN specificity, hydrostatic/SAR contributions and inference efficiency are documented in [discussion experiments](docs/discuss/README.md). They use separate configurations and are not added to the default experiment catalog.
