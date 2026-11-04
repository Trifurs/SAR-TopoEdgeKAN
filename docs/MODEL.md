# SAR-TopoEdgeKAN architecture

SAR-TopoEdgeKAN reconstructs flood depth from temporal Sentinel-1 observations, DSM, slope and a supplied flood range. Its main stages are:

1. **Joint temporal SAR pyramid.** Pre-event and event-period VV/VH, change features, internal signed/absolute temporal differences, the supplied range and validity maps feed one encoder with widths 24/48/96/144. SAR features are aligned to event support before the context bottleneck.
2. **Hierarchical terrain pyramid.** Normalized DSM/slope and mask-aware terrain features provide a separate multi-scale branch. A DSM low-envelope proxy supplies terrain context; it is not a measured bare-earth DTM.
3. **Topographic edge KAN.** At stride 8, each node connects to its eight neighbours. The six bounded edge descriptors are signed elevation gradient, absolute elevation gradient, relative height, DSM path barrier, local relief and neighbour distance. Two 6-to-1 KAN heads learn corrections to topographic affinities. Observation support and latent-feature compatibility gate the messages. A bounded signed residual is zero-initialized, so the graph initially preserves the bottleneck features.
4. **Dual-gated decoder.** Independent validity-aware SAR and terrain gates combine additive multi-scale skips with the graph-refined bottleneck. Temporal-change evidence also enters the full-resolution reconstruction path.
5. **Prior-guided depth head.** A componentwise inner-boundary DSM quantile defines a water-surface reference. The head predicts a bounded multiplicative log-depth correction to the resulting depth prior. The final depth is zero outside the supplied range.

For each valid range component, the prior uses boundary quantile 0.25 and minimum depth 0.10 m. With decoder output logit $r$, the full-model mapping is

$$
d_0 = \max\!\left(Q_{0.25}(z_{\mathrm{boundary}})+0.10-z_{\mathrm{DSM}},\;0.10\right),
\qquad
\hat d = M\,d_0\exp\!\left(5\tanh(r/5)\right),
$$

where $M$ is the supplied binary range. The output projection is zero-initialized, giving the prior itself at initialization. The prior and learned residual are evaluated on the same grid as the observations.

The KAN operates on **graph edges**, not as a generic replacement for convolution layers. The graph is a learned topographic aggregation mechanism, not a hydraulic solver. The input contract excludes Sentinel-2 and target depth. The binary flood range is supplied to every method and is not a predicted output.

The default full model has **1,446,262 trainable parameters**. The formal ablation, `configs/ablation/sar_topo_edge_kan_wo_tae_kan.xml`, physically removes the topographic graph and its coupled latent-compatibility mechanism, leaving **1,435,768 trainable parameters**. It retains the same data split, objective, optimizer, schedule, seed, and evaluation protocol. The separate edge-function controls distinguish graph removal from replacing only the KAN edge function.

The shared training objective is specified in the [comparison protocol](COMPARISON_PROTOCOL.md#shared-depth-objective). The [discussion experiments](discuss/README.md) retain matched-input comparators, graph/edge-function controls, and prior/SAR controls.
