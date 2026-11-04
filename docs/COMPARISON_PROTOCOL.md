# Comparison protocol

The trainable catalog contains SAR-TopoEdgeKAN, the edge-KAN removal ablation, DLSIM Attention U-Net, DLSIM LinkNet, ResNet18 depth regression, U-Net depth regression, and U-Net++ depth regression. FwDET v2, TSA, and FlDepth are deterministic terrain methods evaluated on the same test split; they have no trainable loss.

Every learned configuration inherits `configs/base/base.xml`, `configs/base/datasets/flooddepthnet_s1_terrain.xml`, and `configs/base/unified_protocol.xml`. All learned methods use the same fixed train/validation/test manifest, valid-depth flood range, task-adaptive output loss, AdamW learning rate and weight decay, 200-epoch maximum, 60-epoch minimum, batch size 12, BF16 setting, cosine schedule, and validation checkpoint rule. The two DLSIM architectures use the same ten-channel S1/terrain/range input schema as the other learned comparators; model architectures may process those channels differently. These are shared training settings; actual observation parity is assessed separately below.

The default ten-channel comparator schema does not include all of the main model's observations: raw terrain and explicit validity fractions are additional inputs to the main architecture. Therefore the default table alone does not establish input-matched architectural superiority. The opt-in [discussion experiments](discuss/README.md) expose the same 19 observation channels to all five comparators and use one shared trainer for the controlled comparison.

Validation data select checkpoints and stop training. The test split is used only for post-training reporting. For repeated runs, `python validate_random_seeds.py 5` uses the same five seeds for all seven learned configurations and writes per-seed and aggregate test metrics. `python test.py runs/sar_topo_edge_kan/train` evaluates completed standalone learned runs and the three traditional methods on the common test domain.

The `valid_depth_mask` comes from the dataset's target support. Therefore all results are **known-range/oracle flood-depth** results, not end-to-end flood-extent prediction results. An independently supplied range would be required for deployment.

Traditional methods use the supplied range and DSM: FwDET v2 allocates an outer-boundary elevation, TSA fits a per-component first-degree water-surface trend, and FlDepth reconstructs from medial-ridge cross sections and bank elevations. They are included as test-only references rather than loss-controlled neural baselines.
