from typing import List, Optional

import torch
from torch.nn import Module


def update_weights_gradnorm(
    model: Module,
    losses: List[torch.Tensor],
    weights: List[float],
) -> List[float]:
    """Update task weights using a simple gradient norm heuristic.

    Parameters
    ----------
    model : torch.nn.Module
        The model used to compute gradients.
    losses : list of torch.Tensor
        Unweighted loss tensors for each task.
    weights : list of float
        Current task weights in the same order as ``losses``.

    Returns
    -------
    list of float
        Updated weights.

    """
    grad_norms = []
    for loss, w in zip(losses, weights):
        model.zero_grad()
        (w * loss).backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().abs().mean()
        grad_norms.append(total_norm)

    grad_norms = torch.tensor(grad_norms)
    mean_norm = grad_norms.mean()
    new_weights = [float(w * (g / mean_norm)) for w, g in zip(weights, grad_norms)]
    # Normalize to keep sum of weights constant
    sum_old = sum(weights)
    sum_new = sum(new_weights)
    new_weights = [w * sum_old / sum_new for w in new_weights]
    return new_weights


def update_weights_relobralo(
    model: Module,
    losses: List[torch.Tensor],
    weights: List[float],
    avg_grad_norms: Optional[torch.Tensor] = None,
    beta: float = 0.9,
) -> tuple[List[float], torch.Tensor]:
    """Update task weights using a simple ReLoBRaLo heuristic.

    Parameters
    ----------
    model : torch.nn.Module
        The model used to compute gradients.
    losses : list of torch.Tensor
        Unweighted loss tensors for each task.
    weights : list of float
        Current task weights in the same order as ``losses``.
    avg_grad_norms : torch.Tensor, optional
        Running averages of gradient norms for each task.
    beta : float, default 0.9
        Smoothing coefficient for the running average.

    Returns
    -------
    tuple of list, torch.Tensor
        Updated weights and updated running averages.

    """
    if avg_grad_norms is None:
        avg_grad_norms = torch.ones(len(losses))

    grad_norms = []
    for loss, w in zip(losses, weights):
        model.zero_grad()
        (w * loss).backward(retain_graph=True)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().abs().mean()
        grad_norms.append(total_norm)

    grad_norms = torch.tensor(grad_norms)
    avg_grad_norms = beta * avg_grad_norms + (1 - beta) * grad_norms
    ratios = grad_norms / (avg_grad_norms + 1e-8)
    new_weights = [float(w * r) for w, r in zip(weights, ratios)]
    sum_old = sum(weights)
    sum_new = sum(new_weights)
    new_weights = [w * sum_old / sum_new for w in new_weights]

    return new_weights, avg_grad_norms
