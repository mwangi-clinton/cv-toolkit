"""Unit tests for the FlowPrune toolkit."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from flowprune import (
    FlowGraph,
    FlowPrune,
    build_attention_graph,
    compress_layers,
    compute_approximation_errors,
    compute_flow_matrix,
    compute_flowprune,
    compute_pairwise_flows,
    compute_query_flow_vector,
    compute_token_flow,
    first_rank_divergence,
    flow_to_heatmap,
    get_tactic,
    heuristic_tactic_0,
    heuristic_tactic_1,
    heuristic_tactic_2,
    heuristic_tactic_3,
    normalize_compact_list,
    overlay_flow_heatmap,
    prune_and_compress_attention,
    prune_edges,
    solve_layer_retention_dp,
    top_k_jaccard_similarity,
)


@pytest.fixture
def sample_attentions():
    torch.manual_seed(42)
    # 6 layers, 10 tokens
    return [torch.softmax(torch.randn(10, 10), dim=-1) for _ in range(6)]


@pytest.fixture
def multihead_attentions():
    torch.manual_seed(42)
    # 6 layers, 4 heads, 10 tokens
    return [torch.softmax(torch.randn(4, 10, 10), dim=-1) for _ in range(6)]


def test_prune_edges(sample_attentions):
    threshold = 0.05
    pruned, stats = prune_edges(sample_attentions, threshold=threshold)
    assert len(pruned) == len(sample_attentions)
    assert stats["total_elements"] == 6 * 10 * 10
    assert stats["pruned_edges"] >= 0
    assert 0.0 <= stats["sparsity"] <= 1.0

    for p in pruned:
        assert (p[p > 0] >= threshold).all()


def test_normalize_compact_list():
    assert normalize_compact_list([]) == []
    assert normalize_compact_list([(2, 2), (3, 1)]) == []
    # Merge overlapping
    merged = normalize_compact_list([(1, 3), (2, 5)])
    assert merged == [(1, 5)]
    # Disjoint
    disjoint = normalize_compact_list([(1, 2), (4, 6)])
    assert disjoint == [(1, 2), (4, 6)]


def test_solve_layer_retention_dp():
    # Test boundary conditions
    assert solve_layer_retention_dp(np.zeros(2), 3, retain_rate=0.5) == []
    assert solve_layer_retention_dp(np.zeros(10), 11, retain_rate=1.0) == []

    # Valid retention DP
    min_count = np.array([5, 2, 8, 1, 9, 3, 4, 7, 2, 6], dtype=np.int32)
    compact_list = solve_layer_retention_dp(min_count, layer_depth=11, retain_rate=0.4)
    assert isinstance(compact_list, list)
    for start, end in compact_list:
        assert start < end
        assert start >= 1
        assert end <= 10


def test_tactics(sample_attentions):
    for tactic_fn in [heuristic_tactic_0, heuristic_tactic_1, heuristic_tactic_2, heuristic_tactic_3]:
        cl, info = tactic_fn(sample_attentions, retain_rate=0.5, seed=42)
        assert isinstance(cl, list)
        assert "min_count" in info

    assert get_tactic(0) is heuristic_tactic_0
    assert get_tactic("walk") is heuristic_tactic_1
    with pytest.raises(ValueError):
        get_tactic("invalid_tactic_name")


def test_compress_layers(sample_attentions):
    compact_list = [(1, 3)]
    compacted, meta = compress_layers(sample_attentions, compact_list=compact_list)
    # Original 6 transitions: 0 kept, 1..3 merged into 1, 4 kept, 5 kept -> total 4 transitions
    assert len(compacted) == 4
    assert meta["original_layer_count"] == 7
    assert meta["compressed_layer_count"] == 5


def test_prune_and_compress_attention(multihead_attentions):
    compacted, meta = prune_and_compress_attention(
        multihead_attentions,
        retain_rate=0.5,
        threshold=1e-4,
        tactic="walk",
        head_fusion="mean",
        seed=42,
    )
    assert len(compacted) <= len(multihead_attentions)
    assert "pruning" in meta
    assert "compression" in meta


def test_graph_and_flow_calculations(sample_attentions):
    graph, num_layers, num_tokens = build_attention_graph(sample_attentions, threshold=0.0)
    assert num_layers == 7
    assert num_tokens == 10
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0

    # Token flow
    flow_00 = compute_token_flow(graph, source_token=0, target_token=0)
    assert flow_00 >= 0.0

    # Query flow vector
    query_vec = compute_query_flow_vector(graph, target_token=0, num_tokens=10, normalize=True)
    assert query_vec.shape == (10,)
    assert pytest.approx(query_vec.sum().item(), rel=1e-3) == 1.0

    # Flow matrix
    mat = compute_flow_matrix(graph, num_tokens=10, source_indices=[0, 1], target_indices=[0, 1])
    assert mat.shape == (2, 2)

    # Pairwise
    pairs = [(0, 0), (1, 1)]
    pair_res = compute_pairwise_flows(graph, pairs)
    assert len(pair_res) == 2


def test_flowgraph_class(sample_attentions):
    fg = FlowGraph(sample_attentions)
    assert fg.num_layers == 7
    assert fg.num_tokens == 10
    assert fg.num_nodes > 0
    assert fg.num_edges > 0

    val = fg.flow(0, 0)
    assert val >= 0.0

    q_flow = fg.query_flow(target_token=0)
    assert q_flow.shape == (10,)


def test_compute_flowprune_api(sample_attentions, multihead_attentions):
    # Single sample vector
    res_vec = compute_flowprune(
        sample_attentions,
        query_token=0,
        retain_rate=0.5,
        threshold=1e-5,
    )
    assert res_vec.shape == (10,)
    assert pytest.approx(res_vec.sum().item(), rel=1e-3) == 1.0

    # Multi-head batched input
    batch_attns = [torch.stack([m, m], dim=0) for m in multihead_attentions]  # [2, 4, 10, 10]
    res_batch = compute_flowprune(
        batch_attns,
        query_token=0,
        retain_rate=0.5,
        threshold=1e-5,
    )
    assert res_batch.shape == (2, 10)


def test_flowprune_hook_module(multihead_attentions):
    class DummyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = nn.Identity()
            self.attn2 = nn.Identity()

        def forward(self, x):
            return x

    model = DummyTransformer()
    fp = FlowPrune(model, attention_layer_name="attn", retain_rate=0.5)

    # Calling directly with precomputed attentions
    res = fp(attentions=multihead_attentions, query_token=0)
    assert res.shape == (10,)
    fp.remove_hooks()


def test_evaluation_utils():
    a = torch.tensor([0.5, 0.3, 0.2, 0.0])
    b = torch.tensor([0.4, 0.3, 0.1, 0.2])

    iou = top_k_jaccard_similarity(a, b, k=2)
    assert 0.0 <= iou <= 1.0

    frd = first_rank_divergence(a, b, normalize=True)
    assert 0.0 <= frd <= 1.0

    errs = compute_approximation_errors(a, b)
    assert "mae" in errs
    assert "relative_error" in errs


def test_heatmap_utilities():
    vec = torch.rand(17)  # 1 [CLS] + 16 patches
    hm = flow_to_heatmap(vec, grid_size=(4, 4), has_cls_token=True, target_size=(32, 32))
    assert hm.shape == (32, 32)
    assert hm.min() >= 0.0 and hm.max() <= 1.0

    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    overlay = overlay_flow_heatmap(dummy_img, hm)
    assert overlay.shape == (32, 32, 3)
