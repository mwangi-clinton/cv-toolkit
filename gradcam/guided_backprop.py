"""
Guided Backpropagation and Guided Grad-CAM helpers.

Guided Backpropagation modifies the ReLU backward pass so that only positive
gradients flowing into positive activations are retained. This produces a
high-resolution visualization of the input pixels that excited the neurons,
which can be fused with Grad-CAM to create Guided Grad-CAM.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def _set_relu_inplace(model: nn.Module, inplace: bool) -> Dict[int, bool]:
    """Set ``inplace`` on every ReLU in ``model``.

    Returns a mapping of module id -> original ``inplace`` value only for
    modules that were actually changed. This lets multiple explainers share
    the same model without fighting over the restoration order.
    """
    original: Dict[int, bool] = {}
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            current = bool(module.inplace)
            if current != inplace:
                original[id(module)] = current
                module.inplace = inplace
    return original


class GuidedBackprop:
    """Compute Guided Backpropagation saliency maps.

    The implementation replaces the backward behavior of every ``ReLU`` in the
    model: gradients are zeroed unless both the incoming gradient and the
    forward activation are positive. This is equivalent to the definition in
    Springenberg et al., "Striving for Simplicity: The All Convolutional Net".
    """

    def __init__(self, model: nn.Module, use_cuda: Optional[bool] = None) -> None:
        """Initialize Guided Backprop.

        Parameters
        ----------
        model : torch.nn.Module
            The model to visualize.
        use_cuda : bool, optional
            Whether to run on GPU. If None, inferred from model parameters.
        """
        self.model = model
        self.use_cuda = use_cuda if use_cuda is not None else next(model.parameters()).is_cuda
        self._forward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_handles: List[torch.utils.hooks.RemovableHandle] = []

        # Storage for forward activations per ReLU module. A stack is used
        # because some architectures (e.g. ResNet Bottlenecks) call the same
        # ReLU module multiple times per forward pass; backward runs in the
        # reverse order, so popping the last activation gives the matching one.
        self._activations: Dict[int, List[torch.Tensor]] = {}

        # Backward hooks are incompatible with inplace ReLU activations, so
        # disable inplace mode while this explainer is attached.
        self._relu_inplace_original = _set_relu_inplace(model, inplace=False)

        self._register_hooks()
        self.model.eval()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on all ReLU modules."""
        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                forward_handle = module.register_forward_hook(self._save_activation)
                backward_handle = module.register_full_backward_hook(self._guided_relu_backward)
                self._forward_handles.append(forward_handle)
                self._backward_handles.append(backward_handle)

    def _save_activation(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """Forward hook: push the output of each ReLU onto its activation stack."""
        self._activations.setdefault(id(module), []).append(output.detach())

    def _guided_relu_backward(
        self,
        module: nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Backward hook: zero gradients unless activation and grad are positive."""
        activation_stack = self._activations.get(id(module))
        grad = grad_output[0]

        if not activation_stack:
            return grad_input

        # Pop the matching forward activation (backward runs in reverse order).
        activation = activation_stack.pop()

        # Guided Backpropagation: keep gradient only where both the
        # incoming gradient and the forward ReLU activation are positive.
        # For ReLU, activation is always >= 0, so we must use `<= 0` here
        # (not `< 0`) to actually mask out the zero-activation positions.
        guided_grad = grad.clone()
        guided_grad[guided_grad < 0] = 0
        guided_grad[activation <= 0] = 0

        return (guided_grad,)

    def forward(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the Guided Backpropagation saliency map.

        Parameters
        ----------
        input_tensor : torch.Tensor
            A single image tensor of shape ``(1, C, H, W)``.
        target_category : int, optional
            Class index to explain. If None, the predicted class is used.

        Returns
        -------
        torch.Tensor
            Saliency map of the same shape as ``input_tensor``.
        """
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError("GuidedBackprop.forward expects a batch of size 1 with shape (1, C, H, W)")

        # Reset any stale activation stacks from a previous incomplete run.
        for stack in self._activations.values():
            stack.clear()

        input_tensor = input_tensor.clone().requires_grad_(True)
        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        output = self.model(input_tensor)

        if target_category is None:
            target_category = int(output.argmax(dim=-1).item())
        self._last_target_category = target_category

        score = output[:, target_category]

        self.model.zero_grad()
        score.backward()

        if input_tensor.grad is None:
            raise RuntimeError("Input gradients were not computed.")

        return input_tensor.grad.detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(input_tensor, target_category)

    def remove_hooks(self) -> None:
        """Remove all registered hooks and restore ReLU inplace settings."""
        for handle in self._forward_handles:
            handle.remove()
        for handle in self._backward_handles:
            handle.remove()
        self._forward_handles.clear()
        self._backward_handles.clear()

        for module in self.model.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)) and id(module) in self._relu_inplace_original:
                module.inplace = self._relu_inplace_original[id(module)]

    @property
    def last_target_category(self) -> Optional[int]:
        """Return the class index used in the most recent forward call."""
        return getattr(self, "_last_target_category", None)
