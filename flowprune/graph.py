"""Graph construction and Maximum Flow calculations for FlowPrune.

Constructs directed acyclic graphs (DAGs) representing transformer information flow
and solves max-flow problems between token representations using NetworkX
(implementing Highest-Level Preflow Push / HLPP and augmenting path algorithms).
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import networkx as nx
import torch


def build_attention_graph(
    attentions: Sequence[torch.Tensor],
    threshold: float = 0.0,
) -> Tuple[nx.DiGraph, int, int]:
    """Build a directed acyclic network flow graph from layer-wise attention matrices.

    Nodes are represented by strings '{layer}-{token_index}', where:
      - '0-i' is token i at the input layer.
      - '{L}-j' is token j at output layer L (where L = len(attentions)).
    Edges connect layer l to layer l + 1 with capacities given by the attention weights.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen] of length L.
        threshold: Capacity cutoff; edges with capacity <= threshold are omitted.

    Returns:
        Tuple[nx.DiGraph, int, int]:
            - Directed graph with edge capacities.
            - Total number of node layers (L + 1).
            - Number of tokens per layer.
    """
    if not attentions:
        raise ValueError("Cannot build graph from empty attentions sequence.")

    num_layers = len(attentions) + 1
    num_tokens = attentions[0].size(0)

    graph = nx.DiGraph()

    for l, mat in enumerate(attentions):
        if mat.dim() != 2:
            raise ValueError(
                f"Expected 2D attention tensor at layer {l}, got shape {mat.shape}"
            )
        # Find non-zero transitions
        mask = mat > threshold
        if not mask.any():
            continue

        rows, cols = torch.nonzero(mask, as_tuple=True)
        weights = mat[rows, cols]

        rows_list = rows.tolist()
        cols_list = cols.tolist()
        weights_list = weights.tolist()

        src_prefix = f"{l}-"
        dst_prefix = f"{l + 1}-"

        for u, v, w in zip(rows_list, cols_list, weights_list):
            graph.add_edge(f"{src_prefix}{u}", f"{dst_prefix}{v}", capacity=float(w))

    return graph, num_layers, num_tokens


def compute_token_flow(
    graph: nx.DiGraph,
    source_token: int,
    target_token: int,
    source_layer: int = 0,
    target_layer: Optional[int] = None,
    flow_func: Optional[Callable] = None,
) -> float:
    """Compute maximum information flow from a source token to a target token.

    Args:
        graph: Directed attention graph.
        source_token: Token index at source layer.
        target_token: Token index at target layer.
        source_layer: Source layer index (default 0 for input layer).
        target_layer: Target layer index (default final layer).
        flow_func: NetworkX flow algorithm function (defaults to preflow_push / HLPP).

    Returns:
        float: Maximum flow value.
    """
    s_node = f"{source_layer}-{source_token}"
    t_layer = target_layer if target_layer is not None else _get_last_layer_index(graph)
    t_node = f"{t_layer}-{target_token}"

    if not graph.has_node(s_node) or not graph.has_node(t_node):
        return 0.0

    try:
        if flow_func is not None:
            return float(nx.maximum_flow_value(graph, s_node, t_node, flow_func=flow_func))
        return float(nx.maximum_flow_value(graph, s_node, t_node))
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        return 0.0


def compute_query_flow_vector(
    graph: nx.DiGraph,
    target_token: int = 0,
    num_tokens: Optional[int] = None,
    source_layer: int = 0,
    target_layer: Optional[int] = None,
    flow_func: Optional[Callable] = None,
    normalize: bool = True,
    workers: Optional[int] = None,
) -> torch.Tensor:
    """Compute maximum flow from every input token to a specific query token.

    Commonly used for Vision Transformers:
    - target_token = 0 computes attribution to the [CLS] classification token.
    - target_token = k computes attribution for dense patch query k.

    Args:
        graph: Directed attention graph.
        target_token: Target sink token index at the final layer.
        num_tokens: Number of sequence tokens. If None, inferred from graph.
        source_layer: Input layer index (default 0).
        target_layer: Output layer index (default last layer).
        flow_func: Flow algorithm function (defaults to preflow_push).
        normalize: If True, normalizes the flow vector so values sum to 1.0.
        workers: Optional number of parallel worker processes.

    Returns:
        torch.Tensor: 1D flow attribution vector of shape [num_tokens].
    """
    t_layer = target_layer if target_layer is not None else _get_last_layer_index(graph)
    t_node = f"{t_layer}-{target_token}"

    if num_tokens is None:
        num_tokens = _infer_num_tokens(graph, source_layer)

    if not graph.has_node(t_node) or num_tokens == 0:
        return torch.zeros(num_tokens, dtype=torch.float32)

    if workers is not None and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        tasks = [(graph, f"{source_layer}-{i}", t_node, flow_func) for i in range(num_tokens)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            flows = list(executor.map(_eval_single_flow, tasks))
    else:
        flows = []
        for i in range(num_tokens):
            s_node = f"{source_layer}-{i}"
            if not graph.has_node(s_node):
                flows.append(0.0)
                continue

            try:
                if flow_func is not None:
                    val = float(nx.maximum_flow_value(graph, s_node, t_node, flow_func=flow_func))
                else:
                    val = float(nx.maximum_flow_value(graph, s_node, t_node))
            except (nx.NetworkXError, nx.NetworkXUnbounded):
                val = 0.0
            flows.append(val)

    flow_tensor = torch.tensor(flows, dtype=torch.float32)
    if normalize:
        total = flow_tensor.sum()
        if total > 0:
            flow_tensor = flow_tensor / total

    return flow_tensor


def _eval_single_flow(args: Tuple) -> float:
    """Helper for parallel flow computation."""
    graph, s_node, t_node, flow_func = args
    if not graph.has_node(s_node):
        return 0.0
    try:
        if flow_func is not None:
            return float(nx.maximum_flow_value(graph, s_node, t_node, flow_func=flow_func))
        return float(nx.maximum_flow_value(graph, s_node, t_node))
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        return 0.0


def compute_flow_matrix(
    graph: nx.DiGraph,
    num_tokens: Optional[int] = None,
    source_layer: int = 0,
    target_layer: Optional[int] = None,
    source_indices: Optional[Sequence[int]] = None,
    target_indices: Optional[Sequence[int]] = None,
    flow_func: Optional[Callable] = None,
    normalize: bool = False,
) -> torch.Tensor:
    """Compute maximum flow matrix between source and target tokens.

    Args:
        graph: Directed attention graph.
        num_tokens: Total sequence length. If None, inferred from graph.
        source_layer: Input layer index (default 0).
        target_layer: Output layer index (default last layer).
        source_indices: Subset of source token indices (default all tokens).
        target_indices: Subset of target token indices (default all tokens).
        flow_func: NetworkX flow function.
        normalize: If True, normalizes each target column to sum to 1.

    Returns:
        torch.Tensor: Matrix of max-flow values of shape [num_sources, num_targets].
    """
    if num_tokens is None:
        num_tokens = _infer_num_tokens(graph, source_layer)

    sources = list(source_indices) if source_indices is not None else list(range(num_tokens))
    targets = list(target_indices) if target_indices is not None else list(range(num_tokens))

    t_layer = target_layer if target_layer is not None else _get_last_layer_index(graph)
    matrix = torch.zeros(len(sources), len(targets), dtype=torch.float32)

    for j_idx, t_tok in enumerate(targets):
        t_node = f"{t_layer}-{t_tok}"
        if not graph.has_node(t_node):
            continue
        for i_idx, s_tok in enumerate(sources):
            s_node = f"{source_layer}-{s_tok}"
            if not graph.has_node(s_node):
                continue
            try:
                if flow_func is not None:
                    val = float(nx.maximum_flow_value(graph, s_node, t_node, flow_func=flow_func))
                else:
                    val = float(nx.maximum_flow_value(graph, s_node, t_node))
            except (nx.NetworkXError, nx.NetworkXUnbounded):
                val = 0.0
            matrix[i_idx, j_idx] = val

    if normalize:
        col_sums = matrix.sum(dim=0, keepdim=True)
        matrix = torch.where(col_sums > 0, matrix / col_sums, matrix)

    return matrix


def compute_pairwise_flows(
    graph: nx.DiGraph,
    pairs: Sequence[Tuple[int, int]],
    source_layer: int = 0,
    target_layer: Optional[int] = None,
    flow_func: Optional[Callable] = None,
) -> List[Tuple[Tuple[int, int], float]]:
    """Compute maximum flow for an explicit list of (source_token, target_token) pairs.

    Args:
        graph: Directed attention graph.
        pairs: List of (source_token_idx, target_token_idx) pairs.
        source_layer: Source layer index (default 0).
        target_layer: Target layer index (default last layer).
        flow_func: NetworkX flow function.

    Returns:
        List[Tuple[Tuple[int, int], float]]: List of ((src, dst), flow_value).
    """
    t_layer = target_layer if target_layer is not None else _get_last_layer_index(graph)
    results = []

    for s_tok, t_tok in pairs:
        s_node = f"{source_layer}-{s_tok}"
        t_node = f"{t_layer}-{t_tok}"
        if not graph.has_node(s_node) or not graph.has_node(t_node):
            results.append(((s_tok, t_tok), 0.0))
            continue
        try:
            if flow_func is not None:
                val = float(nx.maximum_flow_value(graph, s_node, t_node, flow_func=flow_func))
            else:
                val = float(nx.maximum_flow_value(graph, s_node, t_node))
        except (nx.NetworkXError, nx.NetworkXUnbounded):
            val = 0.0
        results.append(((s_tok, t_tok), val))

    return results


class FlowGraph:
    """Object-oriented wrapper around the pruned/compressed Attention Flow DAG."""

    def __init__(
        self,
        attentions: Sequence[torch.Tensor],
        threshold: float = 0.0,
        flow_func: Optional[Callable] = None,
    ):
        self.raw_attentions = list(attentions)
        self.threshold = threshold
        self.flow_func = flow_func
        self.graph, self.num_layers, self.num_tokens = build_attention_graph(
            self.raw_attentions, threshold=threshold
        )

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def last_layer(self) -> int:
        return self.num_layers - 1

    def flow(
        self,
        source_token: int,
        target_token: int,
        source_layer: int = 0,
        target_layer: Optional[int] = None,
    ) -> float:
        """Compute max-flow between source_token and target_token."""
        return compute_token_flow(
            self.graph,
            source_token=source_token,
            target_token=target_token,
            source_layer=source_layer,
            target_layer=target_layer or self.last_layer,
            flow_func=self.flow_func,
        )

    def query_flow(
        self,
        target_token: int = 0,
        normalize: bool = True,
        workers: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute flow from all input tokens to query target_token (e.g. [CLS])."""
        return compute_query_flow_vector(
            self.graph,
            target_token=target_token,
            num_tokens=self.num_tokens,
            source_layer=0,
            target_layer=self.last_layer,
            flow_func=self.flow_func,
            normalize=normalize,
            workers=workers,
        )

    def matrix(
        self,
        source_indices: Optional[Sequence[int]] = None,
        target_indices: Optional[Sequence[int]] = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Compute full or sub-matrix of attention flows."""
        return compute_flow_matrix(
            self.graph,
            num_tokens=self.num_tokens,
            source_layer=0,
            target_layer=self.last_layer,
            source_indices=source_indices,
            target_indices=target_indices,
            flow_func=self.flow_func,
            normalize=normalize,
        )

    def __repr__(self) -> str:
        return (
            f"FlowGraph(layers={self.num_layers}, tokens={self.num_tokens}, "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )


def _get_last_layer_index(graph: nx.DiGraph) -> int:
    """Find maximum layer index present in graph nodes."""
    max_layer = 0
    for node in graph.nodes():
        try:
            layer = int(node.split("-")[0])
            if layer > max_layer:
                max_layer = layer
        except (ValueError, IndexError):
            pass
    return max_layer


def _infer_num_tokens(graph: nx.DiGraph, layer: int = 0) -> int:
    """Infer token count at specified layer from graph nodes."""
    tokens = set()
    prefix = f"{layer}-"
    for node in graph.nodes():
        if node.startswith(prefix):
            try:
                tokens.add(int(node[len(prefix):]))
            except ValueError:
                pass
    return max(tokens) + 1 if tokens else 0
