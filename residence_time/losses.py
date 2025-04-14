import torch


def pinn_loss(
    model,
    X: torch.Tensor,
    Gamma_EI: torch.Tensor,
    W: torch.Tensor,
    tau_I_to_X_target: torch.Tensor = None,
    lambda_phys: float = 1.0,
    lambda_sup: float = 1.0,
) -> torch.Tensor:
    """
    Computes the total loss for a physics-informed neural network (PINN),
    combining a physics-based residual loss and an optional supervised loss.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model that returns (tau_pred, D_pred) given input X.

    X : torch.Tensor [N, D]
        Input features for the model (e.g., lat, alt, time, source, Gamma).

    Gamma_EI : torch.Tensor [N,]
        Mean age of air from entry to interior (Γ_EI), used in the physics-based
        equation for residence time: τ_R = 2·D² / Γ_EI.

    W : torch.Tensor [N,]
        Weighting for the physics loss (e.g., 1/std²). Must be non-negative.

    tau_I_to_X_target : torch.Tensor [N,], optional
        Supervised target for τ_I→X (residence time from interior to exit).
        Used to guide the model using inverse estimates or observational data.
        If not provided, only the physics loss is used.

    lambda_phys : float
        Weight for the physics-informed loss.

    lambda_sup : float
        Weight for the supervised τ_I→X loss (ignored if tau_I_to_X_target is None).

    Returns
    -------
    torch.Tensor
        Scalar loss value (weighted sum of physics + supervised losses).
    """
    tau_pred, D_pred = model(X)

    Gamma_EI = Gamma_EI.clamp(min=1e-4)
    D_clamped = D_pred.clamp(max=100.0)

    # Physics-based τ_R from mean age of air and diffusivity
    tau_phys = 2 * D_clamped**2 / Gamma_EI
    loss_phys = torch.mean(W * (tau_pred - tau_phys)**2)

    # Supervised loss for τ_I→X (interior to exit residence time)
    if tau_I_to_X_target is not None:
        loss_sup = torch.mean((tau_pred - tau_I_to_X_target)**2)
    else:
        loss_sup = torch.tensor(0.0, device=X.device)

    total_loss = lambda_phys * loss_phys + lambda_sup * loss_sup
    return total_loss
