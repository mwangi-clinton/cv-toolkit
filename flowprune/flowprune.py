"""FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network.

Core implementation based on:
    "FlowPrune: Accelerating Attention Flow Calculation by Pruning Flow Network"
    Shuo Xu, Yu Chen, Shuxia Lin, Xin Geng, Xu Yang.
    NeurIPS 2025.
    Code: https://github.com/ATMxsp01/FlowPrune
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

try:
    from .graph import FlowGraph, build_attention_graph, compute_flow_matrix, compute_query_flow_vector
    from .tactics import CompactList, get_tactic, normalize_compact_list
except (ImportError, ValueError):
    from graph import FlowGraph, build_attention_graph, compute_flow_matrix, compute_query_flow_vector
    from tactics import CompactList, get_tactic, normalize_compact_list


def prune_edges(
    attentions: Sequence[torch.Tensor],
    threshold: float = 1e-6,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Prune low-capacity attention edges below threshold theta.

    Based on the Max-Flow Min-Cut Theorem: edges with negligible capacity
    contribute negligibly to global information propagation and can be safely removed.

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        threshold: Edge capacity pruning threshold theta (default 1e-6).

    Returns:
        Tuple[List[torch.Tensor], Dict[str, Any]]:
            - Pruned attention matrices (cloned).
            - Pruning statistics (total edges, cut edges, sparsity).
    """
    pruned = []
    total_edges = 0
    cut_edges = 0

    for mat in attentions:
        m = mat.clone()
        total_edges += m.numel()
        mask = m < threshold
        cut_edges += int(mask.sum().item())
        m[mask] = 0.0
        pruned.append(m)

    sparsity = float(cut_edges) / float(max(1, total_edges))
    stats = {
        "total_elements": total_edges,
        "pruned_edges": cut_edges,
        "sparsity": sparsity,
        "threshold": threshold,
    }
    return pruned, stats


def compress_layers(
    attentions: Sequence[torch.Tensor],
    compact_list: Optional[CompactList] = None,
    threshold: float = 0.0,
    max_capacity: float = 1.0,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Compress non-critical layers and reconnect remaining layers via Boolean matrix multiplication.

    For each interval (start_idx, end_idx) in compact_list, the intermediate
    layers are compressed into a single transition matrix representing multi-hop
    reachability, with capacities set to max_capacity (default 1.0, guaranteed
    by max-flow theory to preserve flow bottlenecks of retained layers).

    Args:
        attentions: Sequence of 2D attention matrices [SeqLen, SeqLen].
        compact_list: Non-overlapping intervals (start, end) of layers to compress.
        threshold: Minimum capacity threshold for reachability.
        max_capacity: Capacity assigned to reconnected transitive edges (default 1.0).

    Returns:
        Tuple[List[torch.Tensor], Dict[str, Any]]:
            - Compressed sequence of attention matrices.
            - Compression metadata.
    """
    compact_list = normalize_compact_list(compact_list) if compact_list else []
    compacted_attention: List[torch.Tensor] = []
    step = 0
    num_original_layers = len(attentions)

    for start_attn, end_attn in compact_list:
        while step < start_attn:
            compacted_attention.append(attentions[step])
            step += 1

        # Multiply connectivity matrices across the compressed interval
        compacted_attn = (attentions[start_attn] > threshold).to(attentions[0].dtype)
        observed_max = float(attentions[start_attn].max().item())

        for step_ in range(start_attn + 1, end_attn + 1):
            next_step = (attentions[step_] > threshold).to(attentions[0].dtype)
            compacted_attn = (torch.mm(compacted_attn, next_step) > 0).to(attentions[0].dtype)
            observed_max = max(observed_max, float(attentions[step_].max().item()))

        cap = max_capacity if max_capacity is not None else max(observed_max, 1.0)
        compacted_attn = compacted_attn * cap
        compacted_attention.append(compacted_attn)
        step = end_attn + 1

    while step < num_original_layers:
        compacted_attention.append(attentions[step])
        step += 1

    num_compressed_layers = len(compacted_attention)
    metadata = {
        "original_layer_count": num_original_layers + 1,
        "compressed_layer_count": num_compressed_layers + 1,
        "compact_list": compact_list,
        "compression_ratio": float(num_compressed_layers) / float(max(1, num_original_layers)),
    }
    return compacted_attention, metadata


def prune_and_compress_attention(
    attentions: Sequence[torch.Tensor],
    retain_rate: float = 0.5,
    threshold: float = 1e-6,
    tactic: Union[int, str] = 1,
    head_fusion: str = "mean",
    max_capacity: float = 1.0,
    seed: Optional[int] = None,
    **tactic_kwargs: Any,
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Complete FlowPrune graph reduction pipeline: fusion, edge pruning, DP, and layer compression.

    Args:
        attentions: Sequence of attention tensors. Each can be:
            - 2D: [SeqLen, SeqLen]
            - 3D: [NumHeads, SeqLen, SeqLen]
        retain_rate: Fraction of layers to retain (e.g. 0.8, 0.5, 0.25).
        threshold: Edge capacity pruning threshold theta (default 1e-6).
        tactic: Heuristic tactic ID (0, 1, 2, 3) or name ('walk', 'permutation', etc.).
        head_fusion: Fusion strategy across attention heads ('mean', 'max', 'min', or int head index).
        max_capacity: Capacity for reconnected edges (default 1.0).
        seed: Optional random seed for reproducible sampling.
        **tactic_kwargs: Additional parameters passed to the tactic function.

    Returns:
        Tuple[List[torch.Tensor], Dict[str, Any]]:
            - Compacted and pruned attention matrices.
            - Complete execution metadata dictionary.
    """
    if not attentions:
        raise ValueError("attentions sequence is empty.")

    # 1. Multi-head fusion
    fused_attns: List[torch.Tensor] = []
    for l_idx, attn in enumerate(attentions):
        if attn.dim() == 3:
            # [NumHeads, SeqLen, SeqLen]
            if isinstance(head_fusion, int):
                fused = attn[head_fusion]
            elif head_fusion == "mean":
                fused = attn.mean(dim=0)
            elif head_fusion == "max":
                fused = attn.max(dim=0)[0]
            elif head_fusion == "min":
                fused = attn.min(dim=0)[0]
            else:
                raise ValueError(f"Unsupported head_fusion '{head_fusion}'")
        elif attn.dim() == 2:
            fused = attn
        else:
            raise ValueError(f"Expected 2D or 3D attention tensor, got shape {attn.shape}")
        fused_attns.append(fused)

    # 2. Edge Pruning
    pruned_attns, prune_stats = prune_edges(fused_attns, threshold=threshold)

    # 3. Dynamic Programming layer selection via sampling tactic
    tactic_fn = get_tactic(tactic)
    compact_list, tactic_stats = tactic_fn(
        pruned_attns,
        retain_rate=retain_rate,
        seed=seed,
        **tactic_kwargs,
    )

    # 4. Layer Compression & Reconnection
    compacted_attns, compact_stats = compress_layers(
        pruned_attns,
        compact_list=compact_list,
        threshold=threshold,
        max_capacity=max_capacity,
    )

    metadata = {
        "pruning": prune_stats,
        "tactic": tactic_stats,
        "compression": compact_stats,
    }
    return compacted_attns, metadata


def compute_flowprune(
    attentions: Sequence[torch.Tensor],
    query_token: Optional[int] = 0,
    retain_rate: float = 0.5,
    threshold: float = 1e-6,
    tactic: Union[int, str] = 1,
    head_fusion: str = "mean",
    flow_func: Optional[Callable] = None,
    normalize_output: bool = True,
    seed: Optional[int] = None,
    workers: Optional[int] = None,
    **tactic_kwargs: Any,
) -> torch.Tensor:
    """Compute attention flow using the FlowPrune framework.

    Supports single-sample and batched attention tensors.

    Args:
        attentions: Sequence of attention tensors across L layers:
            - Single sample: [NumHeads, SeqLen, SeqLen] or [SeqLen, SeqLen]
            - Batched: [Batch, NumHeads, SeqLen, SeqLen]
        query_token: Target token index at final layer to compute attribution for:
            - 0: Typically [CLS] token in classification ViTs.
            - k: Specific spatial patch query token.
            - None: Compute full pairwise token flow matrix [SeqLen, SeqLen].
        retain_rate: Fraction of intermediate layers to retain in (0, 1].
        threshold: Edge pruning threshold theta (default 1e-6).
        tactic: Heuristic sampling tactic (default 1 / 'walk').
        head_fusion: Multi-head reduction ('mean', 'max', 'min', or int head).
        flow_func: NetworkX max-flow solver (default preflow_push / HLPP).
        normalize_output: Whether to normalize output flow vector/columns to sum to 1.
        seed: Random seed for path sampling.
        workers: Optional number of worker processes for parallel flow solving.
        **tactic_kwargs: Extra args for sampling tactic.

    Returns:
        torch.Tensor:
            - If query_token is not None:
                [SeqLen] for single sample, or [Batch, SeqLen] for batch.
            - If query_token is None:
                [SeqLen, SeqLen] for single sample, or [Batch, SeqLen, SeqLen] for batch.
    """
    if not attentions:
        raise ValueError("attentions sequence is empty.")

    first = attentions[0]
    is_batched = (first.dim() == 4)

    if is_batched:
        batch_size = first.size(0)
        batch_results = []
        for b in range(batch_size):
            sample_attns = [a[b] for a in attentions]
            res = _compute_flowprune_single(
                sample_attns,
                query_token=query_token,
                retain_rate=retain_rate,
                threshold=threshold,
                tactic=tactic,
                head_fusion=head_fusion,
                flow_func=flow_func,
                normalize_output=normalize_output,
                seed=seed,
                workers=workers,
                **tactic_kwargs,
            )
            batch_results.append(res)
        return torch.stack(batch_results, dim=0)

    return _compute_flowprune_single(
        attentions,
        query_token=query_token,
        retain_rate=retain_rate,
        threshold=threshold,
        tactic=tactic,
        head_fusion=head_fusion,
        flow_func=flow_func,
        normalize_output=normalize_output,
        seed=seed,
        workers=workers,
        **tactic_kwargs,
    )


def _compute_flowprune_single(
    attentions: Sequence[torch.Tensor],
    query_token: Optional[int],
    retain_rate: float,
    threshold: float,
    tactic: Union[int, str],
    head_fusion: str,
    flow_func: Optional[Callable],
    normalize_output: bool,
    seed: Optional[int],
    workers: Optional[int] = None,
    **tactic_kwargs: Any,
) -> torch.Tensor:
    """Internal helper to compute FlowPrune for a single input sample."""
    compacted, _ = prune_and_compress_attention(
        attentions,
        retain_rate=retain_rate,
        threshold=threshold,
        tactic=tactic,
        head_fusion=head_fusion,
        seed=seed,
        **tactic_kwargs,
    )

    flow_graph = FlowGraph(compacted, threshold=0.0, flow_func=flow_func)

    if query_token is not None:
        return flow_graph.query_flow(
            target_token=query_token, normalize=normalize_output, workers=workers
        )

    return flow_graph.matrix(normalize=normalize_output)


class FlowPrune:
    """High-level FlowPrune manager with PyTorch model hook integration.

    Attaches forward hooks to multi-head self-attention modules of a transformer
    model (e.g. ViT, DeiT, Swin, HuggingFace transformers), extracts attention
    weights during inference, and computes FlowPrune explanations.

    Args:
        model: PyTorch transformer model.
        attention_layer_name: Substring identifying self-attention layers (default 'attn').
        retain_rate: Fraction of layers to retain (default 0.5).
        threshold: Edge capacity pruning threshold (default 1e-6).
        tactic: Heuristic sampling tactic (default 1 / 'walk').
        head_fusion: Multi-head fusion ('mean', 'max', 'min', or int).
        flow_func: Optional NetworkX flow function.
        workers: Optional number of worker processes for parallel flow solving.
        device: Device to run inference on.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_layer_name: str = "attn",
        retain_rate: float = 0.5,
        threshold: float = 1e-6,
        tactic: Union[int, str] = 1,
        head_fusion: str = "mean",
        flow_func: Optional[Callable] = None,
        workers: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.attention_layer_name = attention_layer_name
        self.retain_rate = retain_rate
        self.threshold = threshold
        self.tactic = tactic
        self.head_fusion = head_fusion
        self.flow_func = flow_func
        self.workers = workers
        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")

        self.attentions: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()
        for name, module in self.model.named_modules():
            if self.attention_layer_name in name.lower():
                h = module.register_forward_hook(self._hook_fn)
                self._hooks.append(h)

    def _hook_fn(self, module, inp, output):
        # HuggingFace / timm often return (attn_output, attn_weights) or tensor
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self.attentions.append(output[1].detach().cpu())
        elif isinstance(output, torch.Tensor) and output.dim() >= 3:
            self.attentions.append(output.detach().cpu())

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def remove_hooks(self):
        """Remove registered forward hooks."""
        self._remove_hooks()

    def __call__(
        self,
        input_tensor: Optional[torch.Tensor] = None,
        query_token: Optional[int] = 0,
        attentions: Optional[Sequence[torch.Tensor]] = None,
        retain_rate: Optional[float] = None,
        threshold: Optional[float] = None,
        tactic: Optional[Union[int, str]] = None,
        normalize_output: bool = True,
        seed: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> torch.Tensor:
        """Run forward pass and compute FlowPrune attention flow.

        Args:
            input_tensor: Model input tensor [B, C, H, W] (if running forward pass).
            query_token: Target sink token (e.g. 0 for [CLS] token, or None for full matrix).
            attentions: Precomputed attention matrices if forward pass was already run.
            retain_rate: Override retain rate.
            threshold: Override pruning threshold.
            tactic: Override sampling tactic.
            normalize_output: Whether to normalize output vector.
            seed: Random seed for sampling.
            workers: Override worker processes for flow computation.

        Returns:
            torch.Tensor: Attribution vector or matrix.
        """
        if attentions is None:
            if input_tensor is None:
                raise ValueError("Must provide either input_tensor or attentions.")
            self.attentions.clear()
            self.model.eval()
            with torch.no_grad():
                _ = self.model(input_tensor.to(self.device))
            attns = self.attentions
        else:
            attns = list(attentions)

        if not attns:
            raise RuntimeError(
                "No attention tensors captured. Ensure the model outputs attention weights "
                "or pass precomputed attentions."
            )

        num_workers = workers if workers is not None else self.workers
        return compute_flowprune(
            attns,
            query_token=query_token,
            retain_rate=retain_rate if retain_rate is not None else self.retain_rate,
            threshold=threshold if threshold is not None else self.threshold,
            tactic=tactic if tactic is not None else self.tactic,
            head_fusion=self.head_fusion,
            flow_func=self.flow_func,
            normalize_output=normalize_output,
            seed=seed,
            workers=num_workers,
        )
