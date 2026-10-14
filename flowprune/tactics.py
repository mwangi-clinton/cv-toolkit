"""Layer compression heuristics and dynamic programming tactics for FlowPrune.

Implements min-cut candidate path sampling tactics and DP layer selection based on:
    "FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network"
    Shuo Xu, Yu Chen, Shuxia Lin, Xin Geng, Xu Yang.
    NeurIPS 2025. https://github.com/ATMxsp01/FlowPrune
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import random
import numpy as np
import torch

CompactList = List[Tuple[int, int]]
TacticResult = Tuple[CompactList, Dict[str, Any]]


def normalize_compact_list(compact_list: CompactList) -> CompactList:
    """Normalize and merge overlapping or adjacent layer compression intervals.

    Args:
        compact_list: List of (start_layer, end_layer) intervals to compact.

    Returns:
        CompactList: Sorted, non-overlapping intervals where start < end.
    """
    if not compact_list:
        return []

    # Filter invalid intervals where start >= end
    valid_intervals = [
        (start, end) for start, end in compact_list if start < end
    ]
    if not valid_intervals:
        return []

    valid_intervals.sort(key=lambda rg: (rg[0], rg[1]))
    merged: CompactList = [valid_intervals[0]]

    for start, end in valid_intervals[1:]:
        last_start, last_end = merged[-1]
        if last_end >= start:
            # Overlapping or contiguous interval -> extend
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def solve_layer_retention_dp(
    min_count: np.ndarray,
    layer_depth: int,
    retain_rate: float,
) -> CompactList:
    """Dynamic programming algorithm for optimal layer retention.

    Solves the DP formulation from Appendix A.4 of the FlowPrune paper:
    Maximize the total number of sampled min-cut bottleneck edges in retained
    layers while reducing the intermediate layers to the target retain rate.
    The first layer (0) and the final layer (layer_depth - 1) are always retained.

    Args:
        min_count: Array of shape [layer_depth - 1] with counts of potential
            min-cut edges located at each transition layer.
        layer_depth: Total number of node levels (number of attention transitions + 1).
        retain_rate: Fraction of intermediate layers to retain in (0, 1].

    Returns:
        CompactList: List of (start_layer, end_layer) index intervals to compress.
    """
    num_transitions = layer_depth - 1
    if num_transitions <= 3 or retain_rate >= 1.0:
        return []

    # Target number of intermediate layers to retain
    retain_tot = max(1, round(retain_rate * num_transitions) - 2)
    if retain_tot >= num_transitions - 2:
        return []

    neg_inf = -32768
    f = np.full((layer_depth, retain_tot + 2, 2), neg_inf, dtype=np.int32)
    f[:, 1, 0] = 0

    if len(min_count) > 1:
        f[1, 2, 1] = int(min_count[1])

    for i in range(2, layer_depth - 1):
        cnt_i = int(min_count[i]) if i < len(min_count) else 0
        for j in range(2, retain_tot + 1):
            f[i][j][0] = max(int(f[i - 1][j][0]), int(f[i - 1][j][1]))
            prev_compressed = int(f[i - 1][j - 2][0]) if j >= 2 else neg_inf
            f[i][j][1] = max(int(f[i - 1][j - 1][1]), prev_compressed) + cnt_i

    uncompact_list = [layer_depth - 1]
    j = retain_tot
    i = layer_depth - 2
    k = 1 if f[i][j][1] > f[i][j][0] else 0

    while j > 1 and i > 0:
        if k:
            uncompact_list.append(i)
            val1 = int(f[i - 1][j - 1][1])
            val0 = int(f[i - 1][j - 2][0]) if j >= 2 else neg_inf
            if val1 > val0:
                j -= 1
            else:
                j -= 2
                k = 0
        else:
            k = 1 if f[i - 1][j][0] < f[i - 1][j][1] else 0
        i -= 1

    uncompact_list.append(0)
    uncompact_list.reverse()

    # Intervals between retained layers that have at least 2 consecutive layers to compact
    compact_list: CompactList = [
        (a + 1, b - 1)
        for a, b in zip(uncompact_list[:-1], uncompact_list[1:])
        if b - a > 2
    ]
    return normalize_compact_list(compact_list)


def heuristic_tactic_0(
    attentions: Sequence[torch.Tensor],
    retain_rate: float = 0.5,
    trying_times: int = 20,
    seed: Optional[int] = None,
) -> TacticResult:
    """Tactic 0: Random token permutation path sampling.

    Samples source-to-sink paths across layers via random token permutations,
    identifies the bottleneck (lowest positive capacity) edge on each path,
    and runs DP to select layers to compress.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        retain_rate: Fraction of layers to retain (0.0 to 1.0).
        trying_times: Multiplier for number of sampled permutations.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[CompactList, Dict[str, Any]]: Compression intervals and metadata.
    """
    if seed is not None:
        np.random.seed(seed)

    layer_depth = len(attentions) + 1
    layer_width = attentions[0].size(0)
    num_transitions = len(attentions)

    routes = np.zeros((layer_width, layer_depth), dtype=np.int32)
    min_count = np.zeros(num_transitions, dtype=np.int32)
    tot_count = 0

    attn_np = [a.detach().cpu().numpy() for a in attentions]

    max_loops = max(1, trying_times * 3)
    target_count = trying_times * layer_width

    for _ in range(max_loops):
        if tot_count >= target_count:
            break
        for i in range(routes.shape[1]):
            routes[:, i] = np.random.permutation(layer_width)

        for route in routes:
            min_val = float(attn_np[0][route[0], route[1]])
            min_p = 0
            valid_path = True
            for i in range(1, num_transitions):
                cand = float(attn_np[i][route[i], route[i + 1]])
                if cand <= 0:
                    valid_path = False
                    break
                if cand < min_val:
                    min_val = cand
                    min_p = i

            if valid_path:
                min_count[min_p] += 1
                tot_count += 1

    compact_list = solve_layer_retention_dp(min_count, layer_depth, retain_rate)
    return compact_list, {
        "tactic": "heuristic_tactic_0",
        "min_count": min_count.tolist(),
        "total_sampled_paths": tot_count,
        "retain_rate": retain_rate,
    }


def heuristic_tactic_1(
    attentions: Sequence[torch.Tensor],
    retain_rate: float = 0.5,
    trying_times: int = 10,
    seed: Optional[int] = None,
) -> TacticResult:
    """Tactic 1: Forward random walk along non-zero attention transitions.

    From each token at layer 0, performs a forward random walk across layers
    following available outgoing edges (proportional or uniform among non-zero),
    detects the minimum-capacity bottleneck edge, and runs DP.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        retain_rate: Fraction of layers to retain (0.0 to 1.0).
        trying_times: Multiplier for number of sampled walks per token.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[CompactList, Dict[str, Any]]: Compression intervals and metadata.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    layer_depth = len(attentions) + 1
    layer_width = attentions[0].size(0)
    num_transitions = len(attentions)

    min_count = np.zeros(num_transitions, dtype=np.int32)
    tot_count = 0

    valid_road_counts = [(attn > 0).sum(dim=1).int() for attn in attentions]
    target_count = trying_times * layer_width

    for _ in range(max(1, trying_times * 2)):
        if tot_count >= target_count:
            break
        for start in range(layer_width):
            min_val = 1.1
            min_p = -1
            pos = start
            for p, (road_count, attn) in enumerate(zip(valid_road_counts, attentions)):
                k_val = int(road_count[pos].item())
                if k_val == 0:
                    min_val = 0.0
                    break

                val, ind = attn[pos].topk(k_val)
                rnd = random.randint(0, k_val - 1)
                edge_val = float(val[rnd].item())
                if edge_val < min_val:
                    min_val = edge_val
                    min_p = p
                pos = int(ind[rnd].item())

            if min_p >= 0:
                min_count[min_p] += 1
                tot_count += 1

    compact_list = solve_layer_retention_dp(min_count, layer_depth, retain_rate)
    return compact_list, {
        "tactic": "heuristic_tactic_1",
        "min_count": min_count.tolist(),
        "total_sampled_paths": tot_count,
        "retain_rate": retain_rate,
    }


def heuristic_tactic_2(
    attentions: Sequence[torch.Tensor],
    retain_rate: float = 0.5,
    trying_times: int = 5,
    seed: Optional[int] = None,
) -> TacticResult:
    """Tactic 2: Bidirectional random walk (forward from layer 0 & backward from layer L).

    Performs forward random walks from source tokens and backward random walks
    from sink tokens to sample potential min-cut edges with balanced coverage.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        retain_rate: Fraction of layers to retain (0.0 to 1.0).
        trying_times: Multiplier for number of bidirectional walks.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple[CompactList, Dict[str, Any]]: Compression intervals and metadata.
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    layer_depth = len(attentions) + 1
    layer_width = attentions[0].size(0)
    num_transitions = len(attentions)

    min_count = np.zeros(num_transitions, dtype=np.int32)
    tot_count = 0

    valid_road_counts = [(attn > 0).sum(dim=1).int() for attn in attentions]
    r_valid_road_counts = [(attn > 0).sum(dim=0).int() for attn in attentions]
    target_count = trying_times * layer_width * 2

    for _ in range(max(1, trying_times * 2)):
        if tot_count >= target_count:
            break

        # Forward passes
        for start in range(layer_width):
            min_val = 1.1
            min_p = -1
            pos = start
            for p, (road_count, attn) in enumerate(zip(valid_road_counts, attentions)):
                k_val = int(road_count[pos].item())
                if k_val == 0:
                    break
                val, ind = attn[pos].topk(k_val)
                rnd = random.randint(0, k_val - 1)
                edge_val = float(val[rnd].item())
                if edge_val < min_val:
                    min_val = edge_val
                    min_p = p
                pos = int(ind[rnd].item())

            if min_p >= 0:
                min_count[min_p] += 1
                tot_count += 1

        # Backward passes
        rev_transitions = list(enumerate(zip(r_valid_road_counts, attentions)))
        rev_transitions.reverse()
        for start in range(layer_width):
            min_val = 1.1
            min_p = -1
            pos = start
            for p, (road_count, attn) in rev_transitions:
                k_val = int(road_count[pos].item())
                if k_val == 0:
                    break
                val, ind = attn[:, pos].topk(k_val)
                rnd = random.randint(0, k_val - 1)
                edge_val = float(val[rnd].item())
                if edge_val < min_val:
                    min_val = edge_val
                    min_p = p
                pos = int(ind[rnd].item())

            if min_p >= 0:
                min_count[min_p] += 1
                tot_count += 1

    compact_list = solve_layer_retention_dp(min_count, layer_depth, retain_rate)
    return compact_list, {
        "tactic": "heuristic_tactic_2",
        "min_count": min_count.tolist(),
        "total_sampled_paths": tot_count,
        "retain_rate": retain_rate,
    }


def heuristic_tactic_3(
    attentions: Sequence[torch.Tensor],
    retain_rate: float = 0.5,
    high_threshold: float = 1e-4,
    **kwargs: Any,
) -> TacticResult:
    """Tactic 3: Histogram count of low-capacity edges.

    Directly counts edges whose capacity is strictly positive but below
    high_threshold (potential bottleneck cut candidates) across each layer.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        retain_rate: Fraction of layers to retain (0.0 to 1.0).
        high_threshold: Upper threshold for bottleneck candidate edges.

    Returns:
        Tuple[CompactList, Dict[str, Any]]: Compression intervals and metadata.
    """
    layer_depth = len(attentions) + 1
    num_transitions = len(attentions)
    min_count = np.zeros(num_transitions, dtype=np.int32)

    for i in range(num_transitions):
        attn = attentions[i]
        mask = (attn > 0) & (attn <= high_threshold)
        min_count[i] = int(mask.sum().item())

    compact_list = solve_layer_retention_dp(min_count, layer_depth, retain_rate)
    return compact_list, {
        "tactic": "heuristic_tactic_3",
        "min_count": min_count.tolist(),
        "high_threshold": high_threshold,
        "retain_rate": retain_rate,
    }


TACTICS_REGISTRY = {
    0: heuristic_tactic_0,
    1: heuristic_tactic_1,
    2: heuristic_tactic_2,
    3: heuristic_tactic_3,
    "permutation": heuristic_tactic_0,
    "walk": heuristic_tactic_1,
    "forward": heuristic_tactic_1,
    "bidirectional": heuristic_tactic_2,
    "histogram": heuristic_tactic_3,
    "threshold": heuristic_tactic_3,
}


def get_tactic(name_or_id: Union[int, str]):
    """Retrieve a tactic sampling function by name or integer ID."""
    if name_or_id in TACTICS_REGISTRY:
        return TACTICS_REGISTRY[name_or_id]
    raise ValueError(
        f"Unknown tactic '{name_or_id}'. Supported tactics: {list(TACTICS_REGISTRY.keys())}"
    )
