from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from agents.disc import Disc
from nsga2_solver import run_surrogate_nsga2
from problem.problem import SUPPORTED_PROBLEMS, make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_gp_surrogates,
    fit_kan_surrogates,
    fit_tabpfn_surrogate,
    KANSurrogateModel,
    predict_with_gp_std,
    surrogate_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DISC-guided surrogate-assisted optimization with 80 LHS init + 40 evolution steps."
    )
    parser.add_argument("--problem", type=str, default="ZDT1", choices=SUPPORTED_PROBLEMS)
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_fe", type=int, default=120)
    parser.add_argument("--init_fe", type=int, default=80)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=80)
    parser.add_argument("--mutation_sigma", type=float, default=0.12)
    parser.add_argument("--logit_scale", type=float, default=5.0)
    parser.add_argument("--disc-checkpoint", type=str, default=None)
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--surrogate_model", type=str, default="gp", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--kan_steps", type=int, default=100)
    parser.add_argument("--kan_hidden_width", type=int, default=64)
    parser.add_argument("--kan_grid", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--plot_path", type=str, default=None)
    args = parser.parse_args()

    if int(args.max_fe) <= int(args.init_fe):
        raise ValueError(f"max_fe must be greater than init_fe, got {args.max_fe} and {args.init_fe}.")
    return args


def set_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def latin_hypercube_sample(
    *,
    n_samples: int,
    dim: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lower_arr = np.asarray(lower, dtype=np.float32).reshape(-1)
    upper_arr = np.asarray(upper, dtype=np.float32).reshape(-1)
    if lower_arr.size == 1:
        lower_arr = np.repeat(lower_arr, int(dim))
    if upper_arr.size == 1:
        upper_arr = np.repeat(upper_arr, int(dim))

    cut = np.linspace(0.0, 1.0, int(n_samples) + 1, dtype=np.float32)
    u = rng.random((int(n_samples), int(dim)), dtype=np.float32)
    points = cut[:-1, None] + u * (cut[1:, None] - cut[:-1, None])

    lhs = np.empty_like(points, dtype=np.float32)
    for j in range(int(dim)):
        lhs[:, j] = points[rng.permutation(int(n_samples)), j]

    return (lower_arr + lhs * (upper_arr - lower_arr)).astype(np.float32)


def build_surrogate(args: argparse.Namespace, archive_x: np.ndarray, archive_y: np.ndarray):
    name = surrogate_model_name(args)
    if name == "gp":
        gp_models = fit_gp_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            seed=int(args.seed),
        )

        class _GPWrapper:
            def __init__(self, models):
                self.models = models

            def predict_mean(self, x: np.ndarray) -> np.ndarray:
                from surrogate.surrogate_model import predict_with_gp_mean

                return predict_with_gp_mean(self.models, x)

            def predict_std(self, x: np.ndarray) -> np.ndarray:
                return predict_with_gp_std(self.models, x)

        return _GPWrapper(gp_models)

    if name == "tabpfn":
        return fit_tabpfn_surrogate(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
        )

    if name == "kan":
        kan_models = fit_kan_surrogates(
            archive_x=archive_x,
            archive_y=archive_y,
            device=str(args.device),
            kan_steps=int(args.kan_steps),
            hidden_width=int(args.kan_hidden_width),
            grid=int(args.kan_grid),
            seed=int(args.seed),
        )
        return KANSurrogateModel(models=kan_models, device=str(args.device))

    raise ValueError(f"Unsupported surrogate_model: {name}")


def surrogate_or_models_for_nsga2(surrogate: Any) -> tuple[Any | None, list[Any] | None]:
    models = getattr(surrogate, "models", None)
    if isinstance(models, list) and len(models) > 0:
        return None, models
    return surrogate, None


def make_nsga2_problem_adapter(problem, n_obj: int):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def predict_surrogate_mean(surrogate: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(
    surrogate: Any,
    x: np.ndarray,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(
    *,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    offspring_x: np.ndarray,
    surrogate: Any,
) -> np.ndarray:
    archive_y = np.asarray(archive_y, dtype=np.float32)

    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)

    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=archive_x,
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=offspring_x,
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def build_disc(
    args: argparse.Namespace,
    *,
    map_location: str,
) -> Disc:
    disc = Disc(
        hidden_dim=int(args.hidden_dim),
        n_heads=int(args.n_heads),
        ff_dim=int(args.ff_dim),
        dropout=float(args.dropout),
        logit_scale=float(args.logit_scale),
    ).to(map_location)
    disc.eval()

    if args.disc_checkpoint and not bool(args.random_model):
        state = torch.load(args.disc_checkpoint, map_location=map_location)
        state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        disc.load_state_dict(state_dict, strict=True)

    return disc


def sample_offspring_index(logits: torch.Tensor) -> tuple[int, np.ndarray]:
    probs = torch.softmax(logits, dim=-1)
    idx = int(torch.distributions.Categorical(probs=probs).sample().item())
    return idx, probs.detach().cpu().numpy().reshape(-1)


@dataclass
class StepRecord:
    step: int
    fe: int
    selected_index: int
    selected_x: list[float]
    surrogate_y: list[float]
    true_y: list[float]
    hv: float
    archive_size: int


def load_true_pareto_front(
    problem_name: str,
    dim: int,
    n_obj: int,
    n_points: int = 400,
) -> np.ndarray | None:
    try:
        from pymoo.problems import get_problem
    except Exception:
        return None

    key = str(problem_name).lower()
    try:
        pymoo_problem = get_problem(key, n_var=int(dim), n_obj=int(n_obj))
    except TypeError:
        try:
            pymoo_problem = get_problem(key, n_var=int(dim))
        except Exception:
            return None
    except Exception:
        return None

    try:
        pareto = pymoo_problem.pareto_front(n_pareto_points=int(n_points))
    except TypeError:
        try:
            pareto = pymoo_problem.pareto_front()
        except Exception:
            return None
    except Exception:
        return None

    if pareto is None:
        return None
    pareto = np.asarray(pareto, dtype=np.float32)
    if pareto.ndim != 2 or pareto.shape[1] < int(n_obj):
        return None
    return pareto[:, : int(n_obj)]


def plot_results(
    *,
    args: argparse.Namespace,
    fe_history: list[int],
    hv_history: list[float],
    archive_y: np.ndarray,
    true_pareto: np.ndarray | None,
) -> str:
    plot_path = args.plot_path
    if plot_path is None:
        plot_path = str(Path("png") / f"test_disc_{args.problem.lower()}_seed{int(args.seed)}.png")
    else:
        plot_path = str(Path(plot_path))

    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)

    archive_front = pareto_front(archive_y)
    n_obj = int(archive_y.shape[1])

    fig = plt.figure(figsize=(12, 5))
    ax_hv = fig.add_subplot(1, 2, 1)
    if n_obj == 3:
        ax_pf = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        ax_pf = fig.add_subplot(1, 2, 2)

    ax_hv.plot(fe_history, hv_history, marker="o", linewidth=1.8, markersize=4)
    ax_hv.set_xlabel("FE")
    ax_hv.set_ylabel("Hypervolume")
    ax_hv.set_title("HV vs FE")
    ax_hv.grid(True, alpha=0.3)

    if n_obj == 2:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], s=18, alpha=0.45, label="Archive")
        ax_pf.scatter(archive_front[:, 0], archive_front[:, 1], s=26, alpha=0.9, label="Archive PF")
        if true_pareto is not None and true_pareto.shape[1] >= 2:
            order = np.argsort(true_pareto[:, 0])
            ax_pf.plot(
                true_pareto[order, 0],
                true_pareto[order, 1],
                linewidth=2.0,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.grid(True, alpha=0.3)
    elif n_obj == 3:
        ax_pf.scatter(archive_y[:, 0], archive_y[:, 1], archive_y[:, 2], s=18, alpha=0.30, label="Archive")
        ax_pf.scatter(
            archive_front[:, 0],
            archive_front[:, 1],
            archive_front[:, 2],
            s=28,
            alpha=0.95,
            label="Archive PF",
        )
        if true_pareto is not None and true_pareto.shape[1] >= 3:
            ax_pf.scatter(
                true_pareto[:, 0],
                true_pareto[:, 1],
                true_pareto[:, 2],
                s=8,
                alpha=0.20,
                label="True PF",
            )
        ax_pf.set_xlabel("f1")
        ax_pf.set_ylabel("f2")
        ax_pf.set_zlabel("f3")
    else:
        raise ValueError(f"plot_results currently supports only 2 or 3 objectives, got n_obj={n_obj}.")

    ax_pf.set_title(f"{args.problem} Archive vs True PF")
    ax_pf.legend()

    fig.tight_layout()
    fig.savefig(plot_file, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return str(plot_file.resolve())


def save_npy_outputs(
    *,
    args: argparse.Namespace,
    archive_x: np.ndarray,
    archive_y: np.ndarray,
    final_front: np.ndarray,
    fe_history: list[int],
    hv_history: list[float],
) -> dict[str, str]:
    out_dir = Path("npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"test_disc_{args.problem.lower()}_seed{int(args.seed)}"

    paths = {
        "archive_x": out_dir / f"{stem}_archive_x.npy",
        "archive_y": out_dir / f"{stem}_archive_y.npy",
        "final_front": out_dir / f"{stem}_final_front.npy",
        "fe_history": out_dir / f"{stem}_fe_history.npy",
        "hv_history": out_dir / f"{stem}_hv_history.npy",
    }

    np.save(paths["archive_x"], np.asarray(archive_x, dtype=np.float32))
    np.save(paths["archive_y"], np.asarray(archive_y, dtype=np.float32))
    np.save(paths["final_front"], np.asarray(final_front, dtype=np.float32))
    np.save(paths["fe_history"], np.asarray(fe_history, dtype=np.int64))
    np.save(paths["hv_history"], np.asarray(hv_history, dtype=np.float64))

    return {key: str(path.resolve()) for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    problem = make_problem(args.problem, dim=int(args.dim))
    n_evo_steps = int(args.max_fe) - int(args.init_fe)

    archive_x = latin_hypercube_sample(
        n_samples=int(args.init_fe),
        dim=int(args.dim),
        lower=problem.lower,
        upper=problem.upper,
        seed=int(args.seed),
    )
    archive_y = np.asarray(problem.evaluate(archive_x), dtype=np.float32)
    n_obj = int(archive_y.shape[1])
    ref_point = get_reference_point(args.problem, n_obj=n_obj)
    nsga2_problem = make_nsga2_problem_adapter(problem, n_obj)
    true_pareto = load_true_pareto_front(args.problem, int(args.dim), n_obj)
    fe_history = [int(args.init_fe)]
    hv_history = [float(hypervolume(archive_y, ref_point))]

    print(f"reference_point = {ref_point.tolist()} (from ref_points_hv.py)")
    print(f"iter 0 | front = {int(pareto_front(archive_y).shape[0])} | HV = {hv_history[-1]:.6f}")

    disc = build_disc(args, map_location=str(args.device))
    history: list[StepRecord] = []

    for step in range(n_evo_steps):
        surrogate = build_surrogate(args, archive_x, archive_y)

        offspring_pop_size = int(args.offspring_size)
        nsga2_surrogate, nsga2_models = surrogate_or_models_for_nsga2(surrogate)
        offspring_x, offspring_pred = run_surrogate_nsga2(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=nsga2_problem,
            archive_x=archive_x,
            pop_size=offspring_pop_size,
            surrogate_nsga_steps=int(args.surrogate_nsga_steps),
            seed=int(args.seed) + step,
        )
        offspring_x = np.asarray(offspring_x, dtype=np.float32)
        offspring_pred = np.asarray(offspring_pred, dtype=np.float32)
        offspring_sigma = build_offspring_sigma(
            archive_x=archive_x,
            archive_y=archive_y,
            offspring_x=offspring_x,
            surrogate=surrogate,
        )

        progress = float(step) / float(max(n_evo_steps - 1, 1))
        with torch.no_grad():
            out = disc(
                x_true=torch.from_numpy(archive_x).to(device=args.device, dtype=torch.float32),
                y_true=torch.from_numpy(archive_y).to(device=args.device, dtype=torch.float32),
                x_sur=torch.from_numpy(offspring_x).to(device=args.device, dtype=torch.float32),
                y_sur=torch.from_numpy(offspring_pred).to(device=args.device, dtype=torch.float32),
                sigma_sur=torch.from_numpy(offspring_sigma).to(device=args.device, dtype=torch.float32),
                progress=progress,
                lower_bound=np.full(int(args.dim), float(problem.lower), dtype=np.float32),
                upper_bound=np.full(int(args.dim), float(problem.upper), dtype=np.float32),
                decode_type="greedy",
            )
            logits = out["logits"].reshape(-1)
        selected_idx, probs = sample_offspring_index(logits)
        selected_x = offspring_x[selected_idx : selected_idx + 1]
        selected_pred = offspring_pred[selected_idx]
        selected_true = np.asarray(problem.evaluate(selected_x), dtype=np.float32)

        archive_x = np.vstack([archive_x, selected_x]).astype(np.float32)
        archive_y = np.vstack([archive_y, selected_true]).astype(np.float32)
        hv = hypervolume(archive_y, ref_point)
        fe = int(args.init_fe) + step + 1
        front_size = int(pareto_front(archive_y).shape[0])
        fe_history.append(fe)
        hv_history.append(float(hv))

        record = StepRecord(
            step=step + 1,
            fe=fe,
            selected_index=selected_idx,
            selected_x=selected_x.reshape(-1).astype(float).tolist(),
            surrogate_y=selected_pred.astype(float).tolist(),
            true_y=selected_true.reshape(-1).astype(float).tolist(),
            hv=float(hv),
            archive_size=int(archive_x.shape[0]),
        )
        history.append(record)

        print(f"iter {record.step} | front = {front_size} | HV = {record.hv:.6f}")

    final_front = pareto_front(archive_y)
    plot_path = plot_results(
        args=args,
        fe_history=fe_history,
        hv_history=hv_history,
        archive_y=archive_y,
        true_pareto=true_pareto,
    )
    npy_paths = save_npy_outputs(
        args=args,
        archive_x=archive_x,
        archive_y=archive_y,
        final_front=final_front,
        fe_history=fe_history,
        hv_history=hv_history,
    )
    summary = {
        "problem": args.problem,
        "dim": int(args.dim),
        "seed": int(args.seed),
        "max_fe": int(args.max_fe),
        "init_fe": int(args.init_fe),
        "evolution_fe": n_evo_steps,
        "surrogate_model": surrogate_model_name(args),
        "disc_checkpoint": args.disc_checkpoint,
        "random_model": bool(args.random_model),
        "reference_point": ref_point.astype(float).tolist(),
        "archive_size": int(archive_x.shape[0]),
        "final_hv": float(hypervolume(archive_y, ref_point)),
        "final_front_size": int(final_front.shape[0]),
        "final_front": final_front.astype(float).tolist(),
        "plot_path": plot_path,
        "npy_paths": npy_paths,
        "history": [asdict(item) for item in history],
    }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k not in {"history", "final_front"}}, indent=2))


if __name__ == "__main__":
    main()
