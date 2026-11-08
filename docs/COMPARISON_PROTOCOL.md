# Comparison protocol

The trainable catalog contains SAR-TopoEdgeKAN, the edge-KAN removal ablation, DLSIM Attention U-Net, DLSIM LinkNet, ResNet18 depth regression, U-Net depth regression, and U-Net++ depth regression. FwDET v2, TSA, and FlDepth are deterministic terrain methods evaluated on the same test split; they have no trainable loss.

Every learned configuration inherits `configs/base/base.xml`, `configs/base/datasets/flooddepthnet_s1_terrain.xml`, and `configs/base/unified_protocol.xml`. All learned methods use the same fixed train/validation/test manifest, valid-depth flood range, task-adaptive output loss, AdamW learning rate and weight decay, 200-epoch maximum, 60-epoch minimum, batch size 12, BF16 setting, cosine schedule, and validation checkpoint rule. The two DLSIM architectures use the same ten-channel S1/terrain/range input schema as the other learned comparators; model architectures may process those channels differently. These are shared training settings; actual observation parity is assessed separately below.

The default ten-channel comparator schema does not include all of the main model's observations: raw terrain and explicit validity fractions are additional inputs to the main architecture. Therefore the default table alone does not establish input-matched architectural superiority. The opt-in [discussion experiments](discuss/README.md) expose the same 19 observation channels to all five comparators and use one shared trainer for the controlled comparison.

Validation data select checkpoints and stop training. The test split is used only for post-training reporting. For repeated runs, `python validate_random_seeds.py 5` uses the same five seeds for all seven learned configurations and writes per-seed and aggregate test metrics. `python test.py runs/sar_topo_edge_kan/train` evaluates completed standalone learned runs and the three traditional methods on the common test domain.

The `valid_depth_mask` comes from the dataset's target support. Therefore all results are **known-range/oracle flood-depth** results, not end-to-end flood-extent prediction results. An independently supplied range would be required for deployment.

Traditional methods use the supplied range and DSM: FwDET v2 allocates an outer-boundary elevation, TSA fits a per-component first-degree water-surface trend, and FlDepth reconstructs from medial-ridge cross sections and bank elevations. They are included as test-only references rather than loss-controlled neural baselines.

## Dataset release

Use the shared `FloodDepthNet.zip` release and its `metadata/training_manifest.csv` as the authoritative fixed split:

| Split | Samples | Source events |
|---|---:|---:|
| Train | 4,084 | 457 |
| Validation | 584 | 65 |
| Test | 1,167 | 130 |

Each sample contains aligned pre-event SAR (two bands), event-period SAR (two bands), SAR change features (three bands), SAR quality data (two bands), DSM/slope (two bands), reconstructed depth (one band), and validity masks (four bands). The mask bands are `valid_depth_mask`, `DEM_valid_mask`, `slope_valid_mask`, and `S1_event_composite_valid_mask`. Preserve the embedded GeoTIFF masks and NoData when computing input and evaluation validity. Labels are satellite-derived reconstructed depths rather than field measurements.

Follow the [dataset setup](../README.md#dataset) before starting experiments. The data-preparation command verifies the distributed files and binds their manifest, band schema and packaged training-only statistics to the runtime contract. It does not fit new normalization statistics. Repeated seeds retain the same sample membership; validation and test samples are excluded from normalization fitting.

Source event IDs are disjoint between the three splits. Event-chain IDs are retained as metadata and may span splits, so these experiments do not establish independence between event chains.

## Shared depth objective

Every learned model uses the same canonical supervision domain,
`valid_depth_mask & output_valid`. The configured objective is

$$
\begin{aligned}
L={}&0.50\left(L_{\mathrm{linear}}+0.03L_{\log}\right)
 +0.15\,\mathrm{MAE}_{\mathrm{pixel}}
 +0.20\,\mathrm{RMSE}_{\mathrm{pixel}}\\
 &+0.25\,\mathrm{MAE}_{\mathrm{event}}
 +0.05\,\mathrm{RMSE}_{\mathrm{event}}
 +0.25\,\mathrm{MAE}_{\mathrm{hierarchical}}\\
 &+0.05\,\mathrm{RMSE}_{\mathrm{depth\ bins}}
 +0.10\,\mathrm{RMSE}_{\mathrm{hierarchical}}
 +0.03L_{\mathrm{event\ bias}}.
\end{aligned}
$$

`L_linear` is pixel-micro Smooth-L1 in metres; `L_log` applies Smooth-L1 to `log1p` depths. Both use transition width 0.50 in their respective coordinates. Event MAE/RMSE give each batch-present event one vote. Depth-bin RMSE averages RMSE over nonempty frozen train-depth strata. Hierarchical MAE averages refined strata within their primary regime, then primary regimes within each event, then events; hierarchical RMSE is the square root of the corresponding hierarchically averaged squared error. The event-bias term applies Smooth-L1, with transition 0.10 m, to each event's signed mean error before averaging events.

The 14 loss configuration keys in `configs/base/unified_protocol.xml` are:

| Key | Value | Role |
|---|---:|---|
| `objective_mode` | `task_adaptive` | Shared objective identifier |
| `depth_huber_beta_m` | 0.50 | Metre-valued Smooth-L1 transition |
| `log_depth_huber_beta` | 0.50 | Log-depth Smooth-L1 transition |
| `lambda_depth` | 0.50 | Weight of the linear plus weighted log-depth term |
| `lambda_log` | 0.03 | Log-depth weight inside the depth term |
| `lambda_metric_pixel_mae` | 0.15 | Pixel MAE weight |
| `lambda_metric_pixel_rmse` | 0.20 | Pixel RMSE weight |
| `lambda_metric_event_mae` | 0.25 | Event MAE weight |
| `lambda_metric_event_rmse` | 0.05 | Event RMSE weight |
| `lambda_metric_event_hierarchical_mae` | 0.25 | Hierarchical MAE weight |
| `lambda_metric_depth_bin_rmse` | 0.05 | Depth-bin RMSE weight |
| `lambda_metric_event_hierarchical_rmse` | 0.10 | Hierarchical RMSE weight |
| `lambda_depth_bias` | 0.03 | Event mean-bias weight |
| `depth_bias_beta_m` | 0.10 | Event-bias Smooth-L1 transition |

Stratification boundaries come only from the training assets. Batch-level objective terms are distinct from full-split evaluation metrics, which accumulate all valid test observations before reporting.
