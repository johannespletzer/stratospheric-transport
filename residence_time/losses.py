def pinn_loss(model, X, Gamma_EI, W, tau_I_to_X_target=None, lambda_phys=1.0, lambda_sup=1.0):

    import torch

    tau_pred, D_pred = model(X)
    
    Gamma_EI = Gamma_EI.clamp(min=1e-4)
    D_clamped = torch.clamp(D_pred, max=10.0)

    # Physics-informed estimate
    tau_phys = 2 * D_clamped**2 / Gamma_EI
    loss_phys = torch.mean(W * (tau_pred - tau_phys)**2)

    if tau_I_to_X_target is not None:
        loss_sup = torch.mean((tau_pred - tau_I_to_X_target)**2)
    else:
        loss_sup = torch.tensor(0.0, device=X.device)

    return lambda_phys * loss_phys + lambda_sup * loss_sup
