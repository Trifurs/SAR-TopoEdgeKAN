# SAR-TopoEdgeKAN architecture

SAR-TopoEdgeKAN uses pre-event, event-time, and change Sentinel-1 features alongside a terrain feature pyramid. A joint SAR encoder and terrain encoder feed multi-scale fusion and a context bottleneck. At encoder stride 8, an eight-neighbour graph builds topographic affinities from signed elevation difference, slope, relative height, path barrier, local relief, and distance. Edge KAN functions convert these physical descriptors into neighbour weights; a bounded, identity-initialized residual injects the graph message into the bottleneck. A nested decoder and conditional-positive depth head produce depth inside the supplied flood range. A known-range hydrostatic prior supplies a terrain-based reference to the head.

The KAN operates on **graph edges**, not as a generic replacement for convolution layers. The graph is a learned topographic aggregation mechanism, not a hydraulic solver. The input contract excludes Sentinel-2 and target depth. The binary flood range is supplied to every method and is not a predicted output.

The formal ablation is `configs/ablation/sar_topo_edge_kan_wo_tae_kan.xml`. It disables the topographic-affinity edge-KAN path while retaining the same data split, objective, optimizer, schedule, seed, and evaluation protocol. Its coupled latent-compatibility mechanism is disabled with the parent graph path.
