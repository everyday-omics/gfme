"""The deep-learning and tabular-foundation-model families for S4.3.

These live apart from `models.py` because they carry weights and a training loop rather than a
handful of hyperparameters, and because torch is only needed when one of them is asked for. Each
class is a scikit-learn estimator --- `fit`, `predict_proba` --- so the nested comparison in step
04 drives them through exactly the same path as CatBoost.

One deliberate difference from the earlier standalone script. There, the epoch was chosen by the
AUROC of the fold the model was then scored on, which lets the network stop at whichever epoch
happened to suit that fold. Here the early-stopping split is carved out of the training portion,
so the scored fold is never read during fitting. It costs the networks a little accuracy and is
the reason their numbers are not directly comparable to the earlier run.
"""
import numpy as np

#: The training recipe is the one used in the earlier comparison, minus the leakage.
EPOCHS, BATCH, PATIENCE, WEIGHT_DECAY, VALIDATION = 100, 256, 15, 1e-4, 0.1


def _torch():
    import torch
    return torch


class _Net:
    """Shared fit/predict for the two architectures; subclasses supply the module."""

    def __init__(self, lr=1e-3, epochs=EPOCHS, batch_size=BATCH, dropout=0.2, random_state=42):
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.dropout, self.random_state = dropout, random_state

    def get_params(self, deep=True):
        return {'lr': self.lr, 'epochs': self.epochs, 'batch_size': self.batch_size,
                'dropout': self.dropout, 'random_state': self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _module(self, d):
        raise NotImplementedError

    def fit(self, X, y):
        torch = _torch()
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        torch.manual_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        index = np.arange(len(y))
        fit_index, stop_index = train_test_split(
            index, test_size=VALIDATION, stratify=y, random_state=self.random_state)

        self.scaler_ = StandardScaler().fit(X[fit_index])
        X_fit = self.scaler_.transform(X[fit_index]).astype(np.float32)
        X_stop = self.scaler_.transform(X[stop_index]).astype(np.float32)
        y_fit, y_stop = y[fit_index], y[stop_index]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_ = device
        model = self._module(X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)
        criterion = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_fit), torch.tensor(y_fit)),
            batch_size=self.batch_size, shuffle=True)

        best, best_state, wait = -np.inf, None, 0
        stop_tensor = torch.tensor(X_stop).to(device)
        for _ in range(self.epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                if torch.isnan(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                score = torch.sigmoid(model(stop_tensor)).cpu().numpy()
            auroc = roc_auc_score(y_stop, np.nan_to_num(score, nan=0.5))
            if auroc > best:
                best, wait = auroc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.stopping_auroc_ = float(best)
        return self

    def predict_proba(self, X):
        torch = _torch()
        X = self.scaler_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            score = torch.sigmoid(
                self.model_(torch.tensor(X).to(self.device_))).cpu().numpy()
        score = np.nan_to_num(score, nan=0.5)
        return np.column_stack([1.0 - score, score])


class FlatMLP(_Net):
    """The fully connected network: 512-256-128, batch-norm on the input, GELU, dropout."""

    def _module(self, d):
        torch = _torch()
        nn = torch.nn

        class Module(nn.Module):
            def __init__(self, d, dropout):
                super().__init__()
                self.net = nn.Sequential(
                    nn.BatchNorm1d(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(128, 1))

            def forward(self, x):
                return self.net(x).squeeze(-1)

        return Module(d, self.dropout)


class SimpleCNN(_Net):
    """A 1D convolution over the feature vector.

    The features have no spatial order --- column 3 is not adjacent to column 4 in any meaningful
    sense --- so this architecture is included as the control it was in the earlier comparison
    rather than because the inductive bias is expected to fit.
    """

    def _module(self, d):
        torch = _torch()
        nn = torch.nn

        class Module(nn.Module):
            def __init__(self, dropout):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(16))
                self.fc = nn.Sequential(nn.Linear(64 * 16, 128), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(128, 1))

            def forward(self, x):
                x = self.conv(x.unsqueeze(1))
                return self.fc(x.flatten(1)).squeeze(-1)

        return Module(self.dropout)


class TabularFM:
    """TabPFN or TabICL behind a PCA, which is how they were run in the earlier comparison.

    Both models cap the number of features they accept, well below the 750 here, so the reduction
    is a requirement of the model rather than a tuning choice. `n_components` is therefore treated
    as the configuration axis for these two families.
    """

    def __init__(self, backend='tabicl', n_components=100, random_state=42):
        self.backend, self.n_components, self.random_state = backend, n_components, random_state

    def get_params(self, deep=True):
        return {'backend': self.backend, 'n_components': self.n_components,
                'random_state': self.random_state}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X)
        self.pca_ = PCA(n_components=min(self.n_components, X.shape[1]),
                        random_state=self.random_state)
        reduced = self.pca_.fit_transform(self.scaler_.transform(X))

        if self.backend == 'tabpfn':
            from tabpfn import TabPFNClassifier
            import torch
            self.model_ = TabPFNClassifier(
                device='cuda' if torch.cuda.is_available() else 'cpu', n_estimators=8)
        elif self.backend == 'tabicl':
            from tabicl import TabICLClassifier
            self.model_ = TabICLClassifier()
        else:
            raise ValueError(f'unknown tabular FM backend {self.backend!r}')

        self.model_.fit(reduced, np.asarray(y))
        return self

    def predict_proba(self, X):
        reduced = self.pca_.transform(self.scaler_.transform(np.asarray(X, dtype=np.float64)))
        return self.model_.predict_proba(reduced)
