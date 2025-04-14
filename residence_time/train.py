
def create_train_val_loaders(X, Gamma, W, tau_R=None, batch_size=256, val_split=0.2, device='cuda'):

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split

    X_train, X_val, Gamma_train, Gamma_val, W_train, W_val, tau_train, tau_val = train_test_split(
        X, Gamma, W, tau_R, test_size=val_split, random_state=42
    )

    def make_loader(X, Gamma, W, tau):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        Gamma_tensor = torch.tensor(Gamma, dtype=torch.float32).unsqueeze(1).to(device)
        W_tensor = torch.tensor(W, dtype=torch.float32).unsqueeze(1).to(device)
        tau_tensor = torch.tensor(tau, dtype=torch.float32).unsqueeze(1).to(device) if tau is not None else None
        dataset = TensorDataset(X_tensor, Gamma_tensor, W_tensor, tau_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return make_loader(X_train, Gamma_train, W_train, tau_train), make_loader(X_val, Gamma_val, W_val, tau_val)

def scale_variables(X_obs):

    from sklearn.preprocessing import MinMaxScaler

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_obs)

    return X_scaled, scaler_X
