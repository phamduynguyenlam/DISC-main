from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination


@dataclass(frozen=True)
class NSGA2Result:
    x: np.ndarray
    y: np.ndarray


def _default_bounds_from_archive(archive_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(archive_x, dtype=np.float32)
    dim = int(x.shape[1]) if x.ndim == 2 else 1
    return np.zeros(dim, dtype=np.float32), np.ones(dim, dtype=np.float32)


def _surrogate_predict_mean(surrogate, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_mean"):
        y = surrogate.predict_mean(x_arr)
    elif hasattr(surrogate, "predict"):
        y = surrogate.predict(x_arr)
    else:
        raise TypeError("surrogate must implement predict_mean(x) or predict(x).")
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim != 2:
        raise ValueError(f"surrogate prediction must return 2D (N, M), got shape={y_arr.shape}.")
    return y_arr


class GPSurrogateProblem(Problem):
    def __init__(self, surrogate, n_var, n_obj, xl, xu):
        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            xl=xl,
            xu=xu,
        )
        self.surrogate = surrogate

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = _surrogate_predict_mean(self.surrogate, np.asarray(X, dtype=np.float32))


class _ModelListSurrogate:
    def __init__(self, models):
        self.models = list(models)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        preds = []
        for model in self.models:
            if hasattr(model, "posterior"):
                model_device = next(model.parameters()).device
                x_tensor = torch.tensor(np.asarray(x_arr, dtype=np.float64), dtype=torch.double, device=model_device)
                with torch.no_grad():
                    mean = model.posterior(x_tensor).mean.detach().cpu().numpy().reshape(-1)
            elif hasattr(model, "predict"):
                mean = np.asarray(model.predict(x_arr), dtype=np.float32).reshape(-1)
            else:
                model_device = next(model.parameters()).device
                x_tensor = torch.tensor(x_arr, dtype=torch.float32, device=model_device)
                with torch.no_grad():
                    mean = model(x_tensor).detach().cpu().numpy().reshape(-1)
            preds.append(np.asarray(mean, dtype=np.float32))
        return np.stack(preds, axis=1).astype(np.float32)


def run_surrogate_nsga2(
    problem,
    archive_x,
    pop_size,
    gps=None,
    surrogate=None,
    surrogate_nsga_steps=100,
    seed=0,
    n_gen=None,
):
    if n_gen is not None:
        surrogate_nsga_steps = n_gen
    if surrogate is None:
        if gps is None:
            raise ValueError("run_surrogate_nsga2 requires either `surrogate` or `gps`.")
        surrogate = _ModelListSurrogate(gps)

    surrogate_problem = GPSurrogateProblem(
        surrogate=surrogate,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
    )

    # Khởi tạo population trực tiếp từ archive hiện tại.
    init_x = np.asarray(archive_x, dtype=np.float64)

    if init_x.shape[0] >= pop_size:
        init_x = init_x[:pop_size].copy()
    else:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, init_x.shape[0], size=pop_size)
        init_x = init_x[idx].copy()

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=init_x,
        eliminate_duplicates=True,
    )

    res = minimize(
        surrogate_problem,
        algorithm,
        termination=get_termination("n_gen", int(surrogate_nsga_steps)),
        seed=seed,
        verbose=False,
        save_history=False,
    )

    return np.asarray(res.X, dtype=np.float64), np.asarray(res.F, dtype=np.float64)
