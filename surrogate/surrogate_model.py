from __future__ import annotations

import contextlib
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

from surrogate.kan import KAN

os.environ["TABPFN_TOKEN"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjoiMWYwZGEyZWEtOGY1Zi00MGNiLWJlNzUtN2U1OTI2YTAxZGFlIiwiZXhwIjoxODA5MDE1MjA3fQ.rFxK90AswdPigPC-vBVUmAELAiVtOvy5YNGfTDUam8A"

def _hydrate_tabpfn_env_from_windows_user_env() -> None:
    if sys.platform != "win32":
        return
    wanted = [key for key in ("TABPFN_TOKEN", "TABPFN_NO_BROWSER") if not os.environ.get(key)]
    if not wanted:
        return
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            for name in wanted:
                try:
                    value, _ = winreg.QueryValueEx(key, name)
                except OSError:
                    continue
                if isinstance(value, str) and value.strip():
                    os.environ[name] = value.strip()
    except Exception:
        pass


_hydrate_tabpfn_env_from_windows_user_env()


def surrogate_model_name(args) -> str:
    return getattr(args, "surrogate_model", getattr(args, "uncertainty_model", "gp"))


def build_dataset(x: np.ndarray, y: np.ndarray, device: str) -> dict[str, torch.Tensor]:
    n = int(x.shape[0])
    n_train = max(2, int(0.8 * n))
    perm = np.random.permutation(n)
    train_id = perm[:n_train]
    test_id = perm[n_train:] if n_train < n else perm[: min(2, n)]
    return {
        "train_input": torch.tensor(x[train_id], dtype=torch.float32, device=device),
        "train_label": torch.tensor(y[train_id], dtype=torch.float32, device=device),
        "test_input": torch.tensor(x[test_id], dtype=torch.float32, device=device),
        "test_label": torch.tensor(y[test_id], dtype=torch.float32, device=device),
    }


class SurrogateModel(ABC):
    @abstractmethod
    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray: ...

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class GPSurrogateModel(SurrogateModel):
    models: list[GaussianProcessRegressor]

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        return predict_with_gp_mean(self.models, x)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        return predict_with_gp_std(self.models, x)


@dataclass
class KANSurrogateModel(SurrogateModel):
    models: list[KAN]
    device: str

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        dev = self.device if device is None else str(device)
        return predict_with_kan(self.models, x, dev)


@dataclass
class TabPFNSurrogateModel(SurrogateModel):
    model: Any

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        return np.asarray(self.model.predict(np.asarray(x, dtype=np.float32)), dtype=np.float32)

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict_std"):
            raise NotImplementedError("TabPFN surrogate wrapper requires predict_std().")
        return np.asarray(self.model.predict_std(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def fit_kan_surrogates(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    device: str,
    kan_steps: int,
    hidden_width: int,
    grid: int,
    seed: int,
) -> list[KAN]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    models: list[KAN] = []
    for obj_id in range(int(archive_y.shape[1])):
        dataset = build_dataset(archive_x, archive_y[:, [obj_id]], device)
        model = KAN(
            width=[archive_x.shape[1], int(hidden_width), 1],
            grid=int(grid),
            k=3,
            seed=int(seed) + int(obj_id),
            device=str(device),
            auto_save=False,
            save_act=False,
        )
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                model.fit(
                    dataset,
                    opt="Adam",
                    steps=int(kan_steps),
                    lr=1e-2,
                    batch=-1,
                    update_grid=False,
                    lamb=0.0,
                    log=1,
                )
        models.append(model)
    return models


def fit_gp_surrogates(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    seed: int,
) -> list[GaussianProcessRegressor]:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    models: list[GaussianProcessRegressor] = []

    for obj_id in range(int(archive_y.shape[1])):
        kernel = (
            ConstantKernel(constant_value=1.0, constant_value_bounds="fixed")
            * RBF(length_scale=0.5, length_scale_bounds="fixed")
            + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            optimizer=None,
            random_state=int(seed) + int(obj_id),
        )
        model.fit(archive_x, archive_y[:, obj_id])
        models.append(model)

    return models


def predict_with_kan(models: Sequence[Any], x: np.ndarray, device: str) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=str(device))
    preds: list[np.ndarray] = []
    for model in models:
        with torch.no_grad():
            pred = model(x_tensor).detach().cpu().numpy().reshape(-1)
        preds.append(np.asarray(pred, dtype=np.float32))
    return np.stack(preds, axis=1).astype(np.float32)


def predict_with_gp_mean(models: Sequence[GaussianProcessRegressor], x: np.ndarray, device: str | None = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    mean_preds: list[np.ndarray] = []
    for model in models:
        mean = model.predict(x_arr)
        mean_preds.append(np.asarray(mean, dtype=np.float32).reshape(-1))
    return np.stack(mean_preds, axis=1).astype(np.float32)


def predict_with_gp_std(models: Sequence[GaussianProcessRegressor], x: np.ndarray, device: str | None = None) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    std_preds: list[np.ndarray] = []
    for model in models:
        _, std = model.predict(x_arr, return_std=True)
        std_preds.append(np.asarray(std, dtype=np.float32).reshape(-1))
    return np.stack(std_preds, axis=1).astype(np.float32) + 1e-6


def estimate_uncertainty(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    archive_pred: np.ndarray,
    offspring_x: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    archive_pred = np.asarray(archive_pred, dtype=np.float32)
    offspring_x = np.asarray(offspring_x, dtype=np.float32)
    residual = np.abs(archive_pred - archive_y)
    n_neighbors = min(int(n_neighbors), int(archive_x.shape[0]))
    dist = np.linalg.norm(offspring_x[:, None, :] - archive_x[None, :, :], axis=-1)
    nn_idx = np.argsort(dist, axis=1)[:, :n_neighbors]
    local_residual = residual[nn_idx]
    return local_residual.mean(axis=1).astype(np.float32) + 1e-6


def init_uncertainty_archive(archive_x: np.ndarray, archive_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(archive_x, dtype=np.float32).copy(),
        np.asarray(archive_y, dtype=np.float32).copy(),
    )


def update_uncertainty_archive(
    *,
    uncertainty_x: np.ndarray,
    uncertainty_y: np.ndarray,
    new_x: np.ndarray,
    new_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uncertainty_x = np.asarray(uncertainty_x, dtype=np.float32)
    uncertainty_y = np.asarray(uncertainty_y, dtype=np.float32)
    new_x = np.asarray(new_x, dtype=np.float32)
    new_y = np.asarray(new_y, dtype=np.float32)

    if new_x.size == 0 or new_y.size == 0:
        return uncertainty_x, uncertainty_y

    if uncertainty_x.size == 0:
        merged_x = new_x
        merged_y = new_y
    else:
        merged_x = np.vstack([uncertainty_x, new_x])
        merged_y = np.vstack([uncertainty_y, new_y])

    unique_indices: list[int] = []
    for i in range(int(merged_x.shape[0])):
        is_duplicate = False
        for j in unique_indices:
            if np.allclose(merged_x[i], merged_x[j]) and np.allclose(merged_y[i], merged_y[j]):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(i)

    idx = np.asarray(unique_indices, dtype=np.int64)
    return merged_x[idx], merged_y[idx]


def fit_tabpfn_surrogate(*, archive_x: np.ndarray, archive_y: np.ndarray, device: str) -> Any:
    archive_x = np.asarray(archive_x, dtype=np.float32)
    archive_y = np.asarray(archive_y, dtype=np.float32)
    return TabPFNMinMaxSurrogate(
        n_objectives=int(archive_y.shape[1]),
        tabpfn_device=str(device),
    ).fit(archive_x, archive_y)


# --- TabPFN bar-distribution surrogate (moved from tabpfn_surrogate.py) ---


def _as_1d_float(arr: np.ndarray | Sequence[float], *, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.ndim != 1 or out.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 elements, got shape={out.shape}.")
    return out


def _validate_bin_edges(bin_edges: np.ndarray) -> None:
    diffs = np.diff(bin_edges)
    if not np.all(np.isfinite(bin_edges)):
        raise ValueError("bin_edges must be finite.")
    if not np.all(diffs > 0):
        raise ValueError("bin_edges must be strictly increasing.")


def discretize_targets_to_bins(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Map continuous targets into bin indices in [0, K-1] using predefined edges."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    k = int(bin_edges.size - 1)
    idx = np.digitize(y, bin_edges[1:-1], right=False).astype(np.int64, copy=False)
    return np.clip(idx, 0, k - 1)


def uniform_bin_edges_from_targets(y: np.ndarray, n_bins: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("Cannot create bin edges from empty targets.")
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}.")
    lo = float(np.min(y))
    hi = float(np.max(y))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Targets must be finite to build uniform bins.")
    if hi <= lo:
        hi = lo + 1e-3
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TabPFNBins:
    """Discretization bins for TabPFN bar-distribution outputs."""

    edges: np.ndarray
    midpoints: np.ndarray

    @classmethod
    def from_edges(cls, edges: np.ndarray | Sequence[float]) -> "TabPFNBins":
        edges_arr = _as_1d_float(edges, name="bin_edges")
        _validate_bin_edges(edges_arr)
        mid = (edges_arr[:-1] + edges_arr[1:]) * 0.5
        return cls(edges=edges_arr, midpoints=mid.astype(np.float32))

    @property
    def k(self) -> int:
        return int(self.midpoints.size)


def tabpfn_probs_to_mean_std(
    probs: np.ndarray,
    bins: TabPFNBins,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert discrete probabilities into mean/std using bin midpoints."""
    p = np.asarray(probs, dtype=np.float32)
    if p.ndim != 2:
        raise ValueError(f"probs must have shape (N, K), got shape={p.shape}.")
    if p.shape[1] != bins.k:
        raise ValueError(f"probs K={p.shape[1]} does not match bins K={bins.k}.")

    if normalize:
        denom = np.maximum(p.sum(axis=1, keepdims=True), float(eps))
        p = p / denom

    mu = bins.midpoints.reshape(1, -1)
    mean = np.sum(p * mu, axis=1)
    var = np.sum(p * (mu - mean.reshape(-1, 1)) ** 2, axis=1)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


class TabPFNObjectiveSurrogate:
    """Single-objective TabPFN surrogate producing (mean, std) via bin probabilities."""

    def __init__(self, model: Any, bin_edges: np.ndarray | Sequence[float]):
        self.model = model
        self.bins = TabPFNBins.from_edges(bin_edges)
        self._fit_classes: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNObjectiveSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(
                f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}."
            )

        y_bins = discretize_targets_to_bins(y_arr, self.bins.edges)
        if not hasattr(self.model, "fit"):
            raise TypeError("Wrapped model does not implement fit().")
        self.model.fit(x_arr, y_bins)

        classes = getattr(self.model, "classes_", None)
        self._fit_classes = None if classes is None else np.asarray(classes, dtype=np.int64).reshape(-1)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        p = np.asarray(self.model.predict_proba(x_arr), dtype=np.float32)
        if p.ndim != 2:
            raise ValueError(f"predict_proba must return (N, K), got shape={p.shape}.")
        if p.shape[1] == self.bins.k:
            return p

        classes = self._fit_classes
        if classes is None:
            raise ValueError(
                f"predict_proba returned K={p.shape[1]} but model has no classes_ to map into K={self.bins.k} bins."
            )

        full = np.zeros((p.shape[0], self.bins.k), dtype=np.float32)
        for col, cls_id in enumerate(classes.tolist()):
            if 0 <= int(cls_id) < self.bins.k:
                full[:, int(cls_id)] = p[:, col]
        return full

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.predict_proba(x)
        return tabpfn_probs_to_mean_std(p, self.bins, normalize=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


class TabPFNSurrogate:
    """Multi-objective TabPFN surrogate (one classifier per objective)."""

    def __init__(self, objective_models: Sequence[Any], bin_edges: np.ndarray | Sequence[float]):
        if not objective_models:
            raise ValueError("objective_models must be a non-empty sequence.")
        self.objectives = [TabPFNObjectiveSurrogate(model=m, bin_edges=bin_edges) for m in objective_models]

    @property
    def n_objectives(self) -> int:
        return int(len(self.objectives))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")

        for obj_idx, obj in enumerate(self.objectives):
            obj.fit(x_arr, y_arr[:, obj_idx])
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        for obj in self.objectives:
            m, s = obj.predict_mean_std(x)
            means.append(m)
            stds.append(s)
        return np.stack(means, axis=1).astype(np.float32), np.stack(stds, axis=1).astype(np.float32)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


def build_tabpfn_surrogate(
    n_objectives: int,
    bin_edges: np.ndarray | Sequence[float],
    *,
    tabpfn_device: str = "cpu",
    use_many_class_extension: bool = False,
    random_state: int | None = 0,
) -> TabPFNSurrogate:
    """Factory helper that constructs TabPFN classifier surrogates (optional dependency)."""
    try:
        from tabpfn import TabPFNClassifier  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("tabpfn is not installed. Install it with `pip install tabpfn`.") from exc

    models: list[Any] = []
    for _ in range(int(n_objectives)):
        base = TabPFNClassifier()
        if use_many_class_extension:
            try:
                from tabpfn_extensions.manyclass_classifier import ManyClassClassifier  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise ImportError("tabpfn-extensions[many_class] required for use_many_class_extension=True.") from exc
            base = ManyClassClassifier(estimator=base, random_state=random_state)
        models.append(base)
    return TabPFNSurrogate(objective_models=models, bin_edges=bin_edges)


class TabPFNMinMaxSurrogate:
    """TabPFN surrogate with per-fit min-max normalization and adaptive bin count K."""

    def __init__(
        self,
        n_objectives: int,
        *,
        tabpfn_device: str = "cpu",
        use_many_class_extension: bool = False,
        random_state: int | None = 0,
    ):
        self.n_objectives = int(n_objectives)
        if self.n_objectives <= 0:
            raise ValueError(f"n_objectives must be positive, got {n_objectives}.")
        self.tabpfn_device = str(tabpfn_device)
        self.use_many_class_extension = bool(use_many_class_extension)
        self.random_state = random_state

        self._x_min: np.ndarray | None = None
        self._x_rng: np.ndarray | None = None
        self._y_min: np.ndarray | None = None
        self._y_rng: np.ndarray | None = None
        self._bins: TabPFNBins | None = None
        self._model: TabPFNSurrogate | None = None

    @staticmethod
    def _minmax_fit(arr: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        min_v = np.min(arr, axis=0).astype(np.float32)
        max_v = np.max(arr, axis=0).astype(np.float32)
        rng = np.maximum(max_v - min_v, float(eps)).astype(np.float32)
        return min_v, rng

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        if self._x_min is None or self._x_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing x stats).")
        return ((np.asarray(x, dtype=np.float32) - self._x_min) / self._x_rng).astype(np.float32)

    def _norm_y(self, y: np.ndarray) -> np.ndarray:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y stats).")
        return ((np.asarray(y, dtype=np.float32) - self._y_min) / self._y_rng).astype(np.float32)

    def _unnorm_y_mean_std(self, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._y_min is None or self._y_rng is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet (missing y stats).")
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        return (self._y_min + mean * self._y_rng).astype(np.float32), (std * self._y_rng).astype(np.float32)

    @staticmethod
    def _choose_k(n_samples: int) -> int:
        n_samples = int(n_samples)
        if n_samples <= 0:
            return 5
        k = int(np.sqrt(float(n_samples)))
        return int(min(20, max(5, k)))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TabPFNMinMaxSurrogate":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 2:
            raise ValueError(f"x must have shape (N, d), got shape={x_arr.shape}.")
        if y_arr.ndim != 2:
            raise ValueError(f"y must have shape (N, m), got shape={y_arr.shape}.")
        if y_arr.shape[0] != x_arr.shape[0]:
            raise ValueError(f"x and y must have the same number of rows, got {x_arr.shape[0]} and {y_arr.shape[0]}.")
        if y_arr.shape[1] != self.n_objectives:
            raise ValueError(f"y must have {self.n_objectives} objectives, got {y_arr.shape[1]}.")

        self._x_min, self._x_rng = self._minmax_fit(x_arr)
        self._y_min, self._y_rng = self._minmax_fit(y_arr)

        x_norm = self._norm_x(x_arr)
        y_norm = self._norm_y(y_arr)

        k = self._choose_k(x_norm.shape[0])
        edges = np.linspace(0.0, 1.0, k + 1, dtype=np.float32)
        self._bins = TabPFNBins.from_edges(edges)
        self._model = build_tabpfn_surrogate(
            n_objectives=self.n_objectives,
            bin_edges=self._bins.edges,
            tabpfn_device=self.tabpfn_device,
            use_many_class_extension=self.use_many_class_extension,
            random_state=self.random_state,
        ).fit(x_norm, y_norm)
        return self

    def predict_mean_std(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("TabPFNMinMaxSurrogate is not fit yet.")
        x_norm = self._norm_x(np.asarray(x, dtype=np.float32))
        mean_norm, std_norm = self._model.predict_mean_std(x_norm)
        return self._unnorm_y_mean_std(mean_norm, std_norm)

    def predict(self, x: np.ndarray) -> np.ndarray:
        mean, _ = self.predict_mean_std(x)
        return mean

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        _, std = self.predict_mean_std(x)
        return std


# Backwards/ergonomic aliases (requested names)
surrogate_model = SurrogateModel
gp = GPSurrogateModel
kan = KANSurrogateModel
tabpfn = TabPFNSurrogateModel
