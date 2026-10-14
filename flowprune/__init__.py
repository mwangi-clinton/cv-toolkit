"""FlowPrune toolkit module.

Accelerating Attention Flow Calculation in Transformers by Pruning Flow Networks
(Xu et al., NeurIPS 2025).

Provides edge capacity pruning, dynamic-programming-based layer compression,
directed acyclic graph construction, and maximum flow computation for explaining
Transformers and Vision Transformers.

Example
-------
>>> import torch
>>> from flowprune import compute_flowprune, FlowPrune
>>> attentions = [torch.randn(8, 197, 197).softmax(dim=-1) for _ in range(12)]
>>> # Compute flow from all input patches to [CLS] token (index 0)
>>> cls_flow = compute_flowprune(attentions, query_token=0, retain_rate=0.25)
>>> print("Flow shape:", cls_flow.shape)
"""

from .flowprune import (
    FlowPrune,
    compress_layers,
    compute_flowprune,
    prune_and_compress_attention,
    prune_edges,
)
from .graph import (
    FlowGraph,
    build_attention_graph,
    compute_flow_matrix,
    compute_pairwise_flows,
    compute_query_flow_vector,
    compute_token_flow,
)
from .tactics import (
    get_tactic,
    heuristic_tactic_0,
    heuristic_tactic_1,
    heuristic_tactic_2,
    heuristic_tactic_3,
    normalize_compact_list,
    solve_layer_retention_dp,
)
from .utils import (
    compute_approximation_errors,
    first_rank_divergence,
    flow_to_heatmap,
    overlay_flow_heatmap,
    plot_flow_attribution_comparison,
    top_k_jaccard_similarity,
)

__all__ = [
    # Core
    "FlowPrune",
    "compute_flowprune",
    "prune_edges",
    "compress_layers",
    "prune_and_compress_attention",
    # Graph & Solvers
    "FlowGraph",
    "build_attention_graph",
    "compute_token_flow",
    "compute_query_flow_vector",
    "compute_flow_matrix",
    "compute_pairwise_flows",
    # Tactics & DP
    "get_tactic",
    "heuristic_tactic_0",
    "heuristic_tactic_1",
    "heuristic_tactic_2",
    "heuristic_tactic_3",
    "normalize_compact_list",
    "solve_layer_retention_dp",
    # Utils & Metrics
    "compute_approximation_errors",
    "first_rank_divergence",
    "flow_to_heatmap",
    "overlay_flow_heatmap",
    "plot_flow_attribution_comparison",
    "top_k_jaccard_similarity",
]
