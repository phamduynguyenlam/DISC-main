import os
import argparse
import copy
import random
import time
from datetime import datetime
from dataclasses import dataclass
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.disc import Disc
from nsga2_solver import run_surrogate_nsga2
from problem.problem import make_problem
from ref_points_hv import get_reference_point
from reward import hypervolume, pareto_front, reward_scheme_1, reward_scheme_2, reward_scheme_3
from surrogate.surrogate_model import (
    estimate_uncertainty,
    fit_gp_surrogates,
    fit_kan_surrogates,
    fit_tabpfn_surrogate,
    KANSurrogateModel,
    predict_with_gp_mean,
    predict_with_gp_std,
)


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@dataclass
class TrainConfig:
    num_workers: int = 12
    episodes_per_worker: int = 1
    max_fe: int = 120
    init_size: int = 80
    batch_size: int = 64
    replay_size: int = 50000
    gamma: float = 1.0
    lr: float = 1e-4
    target_update_interval: int = 20
    train_iters: int = 50
    updates_per_epoch: int = 80
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_iters: int = 10
    hidden_dim: int = 128
    n_heads: int = 8
    ff_dim: int = 256
    dropout: float = 0.0
    logit_scale: float = 1.0
    surrogate_model: str = "kan"
    surrogate_nsga_steps: int = 100
    offspring_size: int = 80
    kan_steps: int = 25
    kan_hidden_width: int = 10
    kan_grid: int = 5
    reward_scheme: int = 1
    policy_mode: str = "epsilon_greedy"
    training_set: int = 1
    heldout_problem: str = "ZDT1"
    weight_dir: str = "weight"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rollout_device: str = "cpu"
    surrogate_device: str = "cpu"


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def extend(self, transitions):
        self.buffer.extend(transitions)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buffer)


def epsilon_by_iter(it, cfg):
    frac = min(1.0, it / cfg.epsilon_decay_iters)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def to_tensor(x, device):
    return torch.tensor(x, dtype=torch.float32, device=device)


def clone_state_dict_cpu(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def select_action_from_output(out):
    ranking = out["ranking"]
    return int(ranking[0, 0].item())


def parse_args():
    parser = argparse.ArgumentParser(description="Train DISC with surrogate-assisted environments.")
    parser.add_argument("--problem", type=str, default="ZDT1")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--reward_scheme", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--surrogate_model", type=str, default="kan", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--training_set", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--updates_per_epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rollout_device", type=str, default="cpu")
    parser.add_argument("--surrogate_device", type=str, default="cpu")
    parser.add_argument("--ray", action="store_true")
    return parser.parse_args()


TRAIN_PROBLEM_POOL = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]


def latin_hypercube_sample(n_samples, dim, lower, upper, seed):
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


def build_surrogate_from_cfg(cfg_dict, archive_x, archive_y):
    surrogate_name = str(cfg_dict.get("surrogate_model", "gp")).lower()
    surrogate_device = str(cfg_dict.get("surrogate_device", cfg_dict.get("device", "cpu")))

    if surrogate_name == "gp":
        gp_models = fit_gp_surrogates(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            seed=int(cfg_dict.get("seed", 0)),
        )

        class _GPWrapper:
            def __init__(self, models):
                self.models = models

            def predict_mean(self, x):
                return predict_with_gp_mean(self.models, x)

            def predict_std(self, x):
                return predict_with_gp_std(self.models, x)

        return _GPWrapper(gp_models)

    if surrogate_name == "kan":
        models = fit_kan_surrogates(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            device=surrogate_device,
            kan_steps=int(cfg_dict.get("kan_steps", 100)),
            hidden_width=int(cfg_dict.get("kan_hidden_width", 64)),
            grid=int(cfg_dict.get("kan_grid", 5)),
            seed=int(cfg_dict.get("seed", 0)),
        )
        return KANSurrogateModel(models=models, device=surrogate_device)

    if surrogate_name == "tabpfn":
        return fit_tabpfn_surrogate(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            device=surrogate_device,
        )

    raise ValueError(f"Unsupported surrogate_model: {surrogate_name}")


def surrogate_or_models_for_nsga2(surrogate):
    models = getattr(surrogate, "models", None)
    if isinstance(models, list) and len(models) > 0:
        return None, models
    return surrogate, None


def make_nsga2_problem_adapter(problem, n_obj):
    class _ProblemAdapter:
        def __init__(self):
            self.n_var = int(problem.dim)
            self.n_obj = int(n_obj)
            self.xl = np.full(int(problem.dim), float(problem.lower), dtype=np.float32)
            self.xu = np.full(int(problem.dim), float(problem.upper), dtype=np.float32)

    return _ProblemAdapter()


def predict_surrogate_mean(surrogate, x):
    return np.asarray(surrogate.predict_mean(np.asarray(x, dtype=np.float32)), dtype=np.float32)


def predict_surrogate_std(surrogate, x):
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(surrogate, "predict_std"):
        try:
            return np.asarray(surrogate.predict_std(x_arr), dtype=np.float32)
        except NotImplementedError:
            pass
    return np.zeros((int(x_arr.shape[0]), 1), dtype=np.float32)


def build_offspring_sigma(archive_x, archive_y, offspring_x, surrogate):
    archive_y = np.asarray(archive_y, dtype=np.float32)
    sigma = predict_surrogate_std(surrogate, offspring_x)
    if sigma.ndim == 1:
        sigma = sigma.reshape(-1, 1)
    if sigma.shape[1] == archive_y.shape[1]:
        return sigma.astype(np.float32)

    archive_pred = predict_surrogate_mean(surrogate, archive_x)
    local_sigma = estimate_uncertainty(
        archive_x=np.asarray(archive_x, dtype=np.float32),
        archive_y=archive_y,
        archive_pred=archive_pred,
        offspring_x=np.asarray(offspring_x, dtype=np.float32),
    )
    if local_sigma.ndim == 1:
        local_sigma = local_sigma.reshape(-1, 1)
    if local_sigma.shape[1] != archive_y.shape[1]:
        local_sigma = np.repeat(local_sigma.mean(axis=1, keepdims=True), archive_y.shape[1], axis=1)
    return local_sigma.astype(np.float32)


def pad_stack_rows(arrays, pad_value=0.0):
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    if len(arrays) == 0:
        raise ValueError("arrays must be non-empty.")

    max_ndim = max(arr.ndim for arr in arrays)
    normalized = []
    for arr in arrays:
        if arr.ndim < max_ndim:
            new_shape = arr.shape + (1,) * (max_ndim - arr.ndim)
            arr = arr.reshape(new_shape)
        normalized.append(arr)

    target_shape = tuple(
        max(int(arr.shape[axis]) for arr in normalized)
        for axis in range(max_ndim)
    )

    padded = []
    for arr in normalized:
        if arr.shape == target_shape:
            padded.append(arr)
            continue

        out = np.full(target_shape, pad_value, dtype=np.float32)
        slices = tuple(slice(0, int(size)) for size in arr.shape)
        out[slices] = arr
        padded.append(out)

    return np.stack(padded, axis=0)


def build_row_mask(arrays):
    arrays = [np.asarray(arr) for arr in arrays]
    max_rows = max(int(arr.shape[0]) for arr in arrays)
    mask = np.zeros((len(arrays), max_rows), dtype=bool)
    for i, arr in enumerate(arrays):
        mask[i, : int(arr.shape[0])] = True
    return mask


def _compute_ddqn_loss_same_objectives(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ) = batch

    device = cfg.device
    archive_mask = torch.as_tensor(build_row_mask(x_true), dtype=torch.bool, device=device)
    candidate_mask = torch.as_tensor(build_row_mask(x_sur), dtype=torch.bool, device=device)
    next_archive_mask = torch.as_tensor(build_row_mask(next_x_true), dtype=torch.bool, device=device)
    next_candidate_mask = torch.as_tensor(build_row_mask(next_x_sur), dtype=torch.bool, device=device)

    x_true = to_tensor(pad_stack_rows(x_true), device)
    y_true = to_tensor(pad_stack_rows(y_true), device)
    x_sur = to_tensor(pad_stack_rows(x_sur), device)
    y_sur = to_tensor(pad_stack_rows(y_sur), device)
    sigma_sur = to_tensor(pad_stack_rows(sigma_sur), device)
    progress = to_tensor(np.asarray(progress).reshape(-1, 1), device)

    lower_bound = to_tensor(pad_stack_rows(lower_bound), device)
    upper_bound = to_tensor(pad_stack_rows(upper_bound), device)

    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    next_x_true = to_tensor(pad_stack_rows(next_x_true), device)
    next_y_true = to_tensor(pad_stack_rows(next_y_true), device)
    next_x_sur = to_tensor(pad_stack_rows(next_x_sur), device)
    next_y_sur = to_tensor(pad_stack_rows(next_y_sur), device)
    next_sigma_sur = to_tensor(pad_stack_rows(next_sigma_sur), device)
    next_progress = to_tensor(np.asarray(next_progress).reshape(-1, 1), device)

    out = agent(
        x_true=x_true,
        y_true=y_true,
        x_sur=x_sur,
        y_sur=y_sur,
        sigma_sur=sigma_sur,
        progress=progress,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        archive_mask=archive_mask,
        candidate_mask=candidate_mask,
        decode_type="q_greedy",
    )

    q_values = out["q_values"]
    q_sa = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_online = agent(
            x_true=next_x_true,
            y_true=next_y_true,
            x_sur=next_x_sur,
            y_sur=next_y_sur,
            sigma_sur=next_sigma_sur,
            progress=next_progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=next_archive_mask,
            candidate_mask=next_candidate_mask,
            decode_type="q_greedy",
        )

        next_actions = torch.argmax(next_online["q_values"], dim=1)

        next_target = target_agent(
            x_true=next_x_true,
            y_true=next_y_true,
            x_sur=next_x_sur,
            y_sur=next_y_sur,
            sigma_sur=next_sigma_sur,
            progress=next_progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=next_archive_mask,
            candidate_mask=next_candidate_mask,
            decode_type="q_greedy",
        )

        next_q = next_target["q_values"].gather(1, next_actions.view(-1, 1)).squeeze(1)
        target = rewards + cfg.gamma * next_q * (1.0 - dones)

    td_error = q_sa - target
    loss = nn.SmoothL1Loss()(q_sa, target)
    metrics = {
        "q_mean": q_sa.detach().mean().item(),
        "q_std": q_sa.detach().std(unbiased=False).item() if q_sa.numel() > 1 else 0.0,
        "target_mean": target.detach().mean().item(),
        "td_error_mean": td_error.detach().mean().item(),
        "td_error_std": td_error.detach().std(unbiased=False).item() if td_error.numel() > 1 else 0.0,
        "reward_mean": rewards.mean().item(),
    }
    return loss, metrics, len(x_true)


def compute_ddqn_loss(agent, target_agent, batch, cfg):
    (
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ) = batch

    objective_counts = [int(np.asarray(arr).shape[1]) for arr in y_true]
    groups = {}
    for idx, n_obj in enumerate(objective_counts):
        groups.setdefault(n_obj, []).append(idx)

    total_count = 0
    total_q_mean = 0.0
    total_q_std = 0.0
    total_target_mean = 0.0
    total_td_error_mean = 0.0
    total_td_error_std = 0.0
    total_r_mean = 0.0
    weighted_loss = None
    group_sizes = []
    group_objectives = []

    batch_items = [
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        actions,
        rewards,
        next_x_true,
        next_y_true,
        next_x_sur,
        next_y_sur,
        next_sigma_sur,
        next_progress,
        dones,
    ]

    for n_obj, indices in groups.items():
        subbatch = []
        for item in batch_items:
            subbatch.append([item[i] for i in indices])

        group_loss, group_metrics, group_count = _compute_ddqn_loss_same_objectives(
            agent=agent,
            target_agent=target_agent,
            batch=tuple(subbatch),
            cfg=cfg,
        )

        total_count += int(group_count)
        group_sizes.append(int(group_count))
        group_objectives.append(int(n_obj))
        total_q_mean += float(group_metrics["q_mean"]) * float(group_count)
        total_q_std += float(group_metrics["q_std"]) * float(group_count)
        total_target_mean += float(group_metrics["target_mean"]) * float(group_count)
        total_td_error_mean += float(group_metrics["td_error_mean"]) * float(group_count)
        total_td_error_std += float(group_metrics["td_error_std"]) * float(group_count)
        total_r_mean += float(group_metrics["reward_mean"]) * float(group_count)

        scaled_loss = group_loss * (float(group_count) / float(len(objective_counts)))
        weighted_loss = scaled_loss if weighted_loss is None else (weighted_loss + scaled_loss)

    if weighted_loss is None or total_count <= 0:
        raise ValueError("Failed to build objective-shape groups for DDQN loss.")

    metrics = {
        "q_mean": total_q_mean / total_count,
        "q_std": total_q_std / total_count,
        "target_mean": total_target_mean / total_count,
        "td_error_mean": total_td_error_mean / total_count,
        "td_error_std": total_td_error_std / total_count,
        "reward_mean": total_r_mean / total_count,
        "shape_group": len(group_sizes),
        "group_sizes": group_sizes,
        "group_objectives": group_objectives,
        "shape_group_detail": {
            int(n_obj): int(sz) for n_obj, sz in zip(group_objectives, group_sizes)
        },
    }
    return weighted_loss, metrics


def compute_env_reward(previous_archive_y, selected_y, ref_point, reward_scheme_id):
    previous_front = pareto_front(np.asarray(previous_archive_y, dtype=np.float32))
    selected_y = np.asarray(selected_y, dtype=np.float32)

    if int(reward_scheme_id) == 1:
        return float(
            reward_scheme_1(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
            )
        )
    if int(reward_scheme_id) == 2:
        return float(
            reward_scheme_2(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
            )
        )
    if int(reward_scheme_id) == 3:
        return float(
            reward_scheme_3(
                previous_front=previous_front,
                selected_objectives=selected_y,
                ref_point=ref_point,
            )
        )
    raise ValueError(f"Unsupported reward_scheme: {reward_scheme_id}")


def build_training_env_specs(problem_name, training_set):
    target_problem = str(problem_name).upper()
    if target_problem not in TRAIN_PROBLEM_POOL:
        raise ValueError(
            f"problem_name must be one of {TRAIN_PROBLEM_POOL} for training-set construction, got {target_problem}."
        )

    if int(training_set) == 1:
        dims = [15, 20, 25]
        problems = [name for name in TRAIN_PROBLEM_POOL if name != target_problem]
    elif int(training_set) == 2:
        dims = [10, 15]
        problems = [name for name in TRAIN_PROBLEM_POOL if name != target_problem]
    elif int(training_set) == 3:
        dims = [15, 20, 25]
        problems = [target_problem]
    else:
        raise ValueError(f"Unsupported training_set: {training_set}. Expected one of {{1, 2, 3}}.")

    env_specs = [{"problem_name": name, "dim": int(dim)} for name in problems for dim in dims]
    if not env_specs:
        raise ValueError("No training environments were created.")
    return env_specs


def env_key(problem_name, dim):
    return f"{str(problem_name).upper()}-{int(dim)}D"


class DiscSAEAEnv:
    def __init__(self, problem_name, dim, seed, cfg_dict):
        self.problem_name = str(problem_name)
        self.dim = int(dim)
        self.seed = int(seed)
        self.cfg = dict(cfg_dict)
        self.problem = make_problem(self.problem_name, dim=self.dim)
        self.lower_bound = np.full(self.dim, float(self.problem.lower), dtype=np.float32)
        self.upper_bound = np.full(self.dim, float(self.problem.upper), dtype=np.float32)
        self.max_steps = max(1, int(self.cfg["max_fe"]) - int(self.cfg["init_size"]))
        self.t = 0
        self.archive_x = None
        self.archive_y = None
        self.offspring_x = None
        self.offspring_y = None
        self.offspring_sigma = None
        self.ref_point = None
        self.nsga2_problem = None
        self.init_hv = None
        self.surrogate = None

    def _progress(self):
        return float(self.t) / float(max(self.max_steps - 1, 1))

    def _surrogate_cfg(self):
        cfg_local = dict(self.cfg)
        cfg_local["seed"] = int(self.seed) + int(self.t)
        return cfg_local

    def _fit_surrogate(self):
        self.surrogate = build_surrogate_from_cfg(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
        )
        return self.surrogate

    def _refresh_offspring(self):
        if self.surrogate is None:
            raise RuntimeError("Surrogate is not initialized. Call _fit_surrogate() before refreshing offspring.")
        surrogate = self.surrogate
        nsga2_surrogate, nsga2_models = surrogate_or_models_for_nsga2(surrogate)
        offspring_x, offspring_y = run_surrogate_nsga2(
            gps=nsga2_models,
            surrogate=nsga2_surrogate,
            problem=self.nsga2_problem,
            archive_x=self.archive_x,
            pop_size=int(self.cfg["offspring_size"]),
            surrogate_nsga_steps=int(self.cfg["surrogate_nsga_steps"]),
            seed=int(self.seed) + int(self.t),
        )
        self.offspring_x = np.asarray(offspring_x, dtype=np.float32)
        self.offspring_y = np.asarray(offspring_y, dtype=np.float32)
        self.offspring_sigma = build_offspring_sigma(
            archive_x=self.archive_x,
            archive_y=self.archive_y,
            offspring_x=self.offspring_x,
            surrogate=surrogate,
        )

    def _build_state(self):
        return {
            "x_true": np.asarray(self.archive_x, dtype=np.float32).copy(),
            "y_true": np.asarray(self.archive_y, dtype=np.float32).copy(),
            "x_sur": np.asarray(self.offspring_x, dtype=np.float32).copy(),
            "y_sur": np.asarray(self.offspring_y, dtype=np.float32).copy(),
            "sigma_sur": np.asarray(self.offspring_sigma, dtype=np.float32).copy(),
            "progress": np.array([self._progress()], dtype=np.float32),
            "lower_bound": self.lower_bound.copy(),
            "upper_bound": self.upper_bound.copy(),
        }

    def reset(self):
        self.t = 0
        self.archive_x = latin_hypercube_sample(
            n_samples=int(self.cfg["init_size"]),
            dim=self.dim,
            lower=self.problem.lower,
            upper=self.problem.upper,
            seed=self.seed,
        )
        self.archive_y = np.asarray(self.problem.evaluate(self.archive_x), dtype=np.float32)
        self.ref_point = np.asarray(
            get_reference_point(self.problem_name, n_obj=int(self.archive_y.shape[1])),
            dtype=np.float32,
        )
        self.init_hv = float(hypervolume(self.archive_y, self.ref_point))
        self.nsga2_problem = make_nsga2_problem_adapter(self.problem, int(self.archive_y.shape[1]))
        # Pretrain surrogate on the initial archive (default: 80 points), then generate first offspring pool.
        self._fit_surrogate()
        self._refresh_offspring()
        return self._build_state()

    def step(self, action_idx, state):
        del state
        chosen_idx = int(np.clip(int(action_idx), 0, int(self.offspring_x.shape[0]) - 1))
        previous_archive_y = np.asarray(self.archive_y, dtype=np.float32).copy()
        chosen_x = self.offspring_x[chosen_idx : chosen_idx + 1]
        chosen_y = np.asarray(self.problem.evaluate(chosen_x), dtype=np.float32)

        self.archive_x = np.vstack([self.archive_x, chosen_x]).astype(np.float32)
        self.archive_y = np.vstack([self.archive_y, chosen_y]).astype(np.float32)

        reward = compute_env_reward(
            previous_archive_y=previous_archive_y,
            selected_y=chosen_y,
            ref_point=self.ref_point,
            reward_scheme_id=int(self.cfg["reward_scheme"]),
        )

        self.t += 1
        done = self.t >= self.max_steps
        # Refit surrogate after admitting the new true-evaluated sample.
        self._fit_surrogate()
        self._refresh_offspring()
        return self._build_state(), float(reward), bool(done)

    def current_hv(self):
        return float(hypervolume(np.asarray(self.archive_y, dtype=np.float32), self.ref_point))


def _rollout_episode_impl(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    device = str(cfg_dict.get("rollout_device", "cpu"))

    agent = Disc(
        hidden_dim=cfg_dict["hidden_dim"],
        n_heads=cfg_dict["n_heads"],
        ff_dim=cfg_dict["ff_dim"],
        dropout=cfg_dict["dropout"],
        logit_scale=cfg_dict["logit_scale"],
        epsilon=epsilon,
    ).to(device)

    agent.load_state_dict(state_dict_cpu)
    agent.eval()

    env = DiscSAEAEnv(
        problem_name=problem_name,
        dim=int(dim),
        seed=int(seed),
        cfg_dict=cfg_dict,
    )
    state = env.reset()
    total_reward = 0.0
    init_hv = float(env.init_hv)
    transitions = []

    done = False
    while not done:
        with torch.no_grad():
            out = agent(
                x_true=to_tensor(state["x_true"][None, ...], device),
                y_true=to_tensor(state["y_true"][None, ...], device),
                x_sur=to_tensor(state["x_sur"][None, ...], device),
                y_sur=to_tensor(state["y_sur"][None, ...], device),
                sigma_sur=to_tensor(state["sigma_sur"][None, ...], device),
                progress=to_tensor(state["progress"].reshape(1, 1), device),
                lower_bound=to_tensor(state["lower_bound"][None, ...], device),
                upper_bound=to_tensor(state["upper_bound"][None, ...], device),
                decode_type=str(cfg_dict.get("policy_mode", "epsilon_greedy")),
                epsilon=epsilon,
            )

        action = select_action_from_output(out)
        next_state, reward, done = env.step(action, state)

        transitions.append((
            state["x_true"],
            state["y_true"],
            state["x_sur"],
            state["y_sur"],
            state["sigma_sur"],
            float(state["progress"][0]),
            state["lower_bound"],
            state["upper_bound"],
            action,
            reward,
            next_state["x_true"],
            next_state["y_true"],
            next_state["x_sur"],
            next_state["y_sur"],
            next_state["sigma_sur"],
            float(next_state["progress"][0]),
            float(done),
        ))

        total_reward += reward
        state = next_state

    return {
        "transitions": transitions,
        "episode_reward": float(total_reward),
        "episode_steps": int(len(transitions)),
        "env_key": env_key(problem_name, dim),
        "init_hv": init_hv,
        "final_hv": float(env.current_hv()),
    }


def rollout_episode_task_local(state_dict_cpu, cfg_dict, problem_name, dim, seed, epsilon):
    return _rollout_episode_impl(
        state_dict_cpu=state_dict_cpu,
        cfg_dict=cfg_dict,
        problem_name=problem_name,
        dim=dim,
        seed=seed,
        epsilon=epsilon,
    )


def save_training_checkpoint(
    agent,
    cfg,
    problem_name,
    epoch,
    mean_reward,
    best_reward,
    best_state_dict=None,
):
    os.makedirs(cfg.weight_dir, exist_ok=True)
    rs_tag = f"rs{int(cfg.reward_scheme)}"
    problem_tag = str(problem_name).lower()

    if mean_reward > best_reward:
        best_path = os.path.join(cfg.weight_dir, f"disc_problem_{problem_tag}_{rs_tag}_best_reward.pth")
        state_dict_to_save = best_state_dict if best_state_dict is not None else agent.state_dict()
        torch.save(
            {
                "epoch": int(epoch),
                "problem_name": str(problem_name),
                "reward_scheme": int(cfg.reward_scheme),
                "mean_reward": float(mean_reward),
                "state_dict": state_dict_to_save,
            },
            best_path,
        )
        best_reward = float(mean_reward)

    if int(epoch) % 5 == 0:
        epoch_path = os.path.join(cfg.weight_dir, f"disc_problem_{problem_tag}_{rs_tag}_epoch_{int(epoch)}.pth")
        torch.save(
            {
                "epoch": int(epoch),
                "problem_name": str(problem_name),
                "reward_scheme": int(cfg.reward_scheme),
                "mean_reward": float(mean_reward),
                "state_dict": agent.state_dict(),
            },
            epoch_path,
        )

    return best_reward


def train_disc_ddqn_ray(
    problem_name="ZDT1",
    dim=30,
    epoch=None,
    reward_scheme=1,
    surrogate_model="kan",
    training_set=1,
    num_workers=None,
    surrogate_nsga_steps=100,
    updates_per_epoch=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cpu",
    use_ray=False,
):
    cfg = TrainConfig()
    if epoch is not None:
        cfg.train_iters = int(epoch)
    cfg.reward_scheme = int(reward_scheme)
    cfg.surrogate_model = str(surrogate_model).lower()
    cfg.training_set = int(training_set)
    cfg.heldout_problem = str(problem_name).upper()
    cfg.surrogate_nsga_steps = int(surrogate_nsga_steps)
    if updates_per_epoch is not None:
        cfg.updates_per_epoch = int(updates_per_epoch)
    if device is not None:
        cfg.device = str(device)
    cfg.rollout_device = str(rollout_device)
    cfg.surrogate_device = str(surrogate_device)
    if num_workers is not None:
        cfg.num_workers = int(num_workers)
    if cfg.surrogate_model not in {"gp", "kan", "tabpfn"}:
        raise ValueError(
            f"Unsupported surrogate_model: {cfg.surrogate_model}. "
            "Expected one of {'gp', 'kan', 'tabpfn'}."
        )
    env_specs = build_training_env_specs(cfg.heldout_problem, cfg.training_set)
    if int(cfg.num_workers) <= 0:
        raise ValueError(f"num_workers must be positive, got {cfg.num_workers}.")
    if int(cfg.train_iters) <= 0:
        raise ValueError(f"epoch must be positive, got {cfg.train_iters}.")
    actual_num_workers = min(int(cfg.num_workers), len(env_specs))
    cfg_dict = cfg.__dict__.copy()
    os.makedirs("training_logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        "training_logs",
        f"trainer_{cfg.heldout_problem.lower()}_set{cfg.training_set}_{ts}.txt",
    )
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg)
        log_fp.write(str(msg) + "\n")

    executor = None
    ray_rollout_remote = None
    ray_mod = None
    if bool(use_ray):
        try:
            import ray as ray_mod  # type: ignore
        except Exception as exc:
            raise ImportError("ray is not available. Install ray or run without --ray.") from exc
        if not ray_mod.is_initialized():
            ray_mod.init(num_cpus=actual_num_workers, ignore_reinit_error=True)
        ray_rollout_remote = ray_mod.remote(num_cpus=1)(rollout_episode_task_local)
    else:
        executor = ProcessPoolExecutor(max_workers=actual_num_workers)

    agent = Disc(
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        logit_scale=cfg.logit_scale,
        epsilon=cfg.epsilon_start,
    ).to(cfg.device)

    target_agent = copy.deepcopy(agent).to(cfg.device)
    target_agent.eval()

    optimizer = optim.Adam(agent.parameters(), lr=cfg.lr)
    best_reward = -float("inf")

    log(
        "Training config | "
        f"heldout={cfg.heldout_problem} | "
        f"training_set={cfg.training_set} | "
        f"envs={len(env_specs)} | "
        f"workers={actual_num_workers} | "
        f"reward_scheme={cfg.reward_scheme} | "
        f"policy={cfg.policy_mode} | "
        f"surrogate={cfg.surrogate_model} | "
        f"sampling_backend={'ray' if use_ray else 'process_pool'} | "
        f"epochs={cfg.train_iters} | "
        f"sur_steps={cfg.surrogate_nsga_steps} | "
        f"updates_per_epoch={cfg.updates_per_epoch} | "
        f"train_device={cfg.device} | "
        f"rollout_device={cfg.rollout_device} | "
        f"surrogate_device={cfg.surrogate_device} | "
        f"lr={cfg.lr:.1e} | "
        f"batch_size={cfg.batch_size} | "
        f"replay_size={cfg.replay_size} | "
        f"gamma={cfg.gamma:.4f} | "
        f"target_update={cfg.target_update_interval} | "
        f"log_path={log_path}"
    )

    for it in range(cfg.train_iters):
        epoch = int(it) + 1
        epsilon = epsilon_by_iter(it, cfg)
        replay = ReplayBuffer(cfg.replay_size)

        log(
            f"[Epoch {epoch:04d}] start | "
            f"set={cfg.training_set} | "
            f"heldout={cfg.heldout_problem} | "
            f"envs_active={len(env_specs)}/{len(env_specs)} | "
            f"surrogate={cfg.surrogate_model} | "
            f"sur_steps={cfg.surrogate_nsga_steps} | "
            f"eps={epsilon:.3f}"
        )

        state_cpu = clone_state_dict_cpu(agent)
        pre_update_state_dict = copy.deepcopy(agent.state_dict())
        futures = []
        for env_idx, spec in enumerate(env_specs):
            for ep in range(int(cfg.episodes_per_worker)):
                seed = 100000 * epoch + 1000 * env_idx + ep
                if bool(use_ray):
                    futures.append(
                        ray_rollout_remote.remote(
                            state_dict_cpu=state_cpu,
                            cfg_dict=cfg_dict,
                            problem_name=spec["problem_name"],
                            dim=int(spec["dim"]),
                            seed=int(seed),
                            epsilon=epsilon,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            rollout_episode_task_local,
                            state_cpu,
                            cfg_dict,
                            spec["problem_name"],
                            int(spec["dim"]),
                            int(seed),
                            epsilon,
                        )
                    )
        if len(futures) == 0:
            raise ValueError("No rollout tasks were created.")
        if bool(use_ray):
            results = ray_mod.get(futures)
        else:
            results = [f.result() for f in futures]

        per_env_stats = {}
        for result in results:
            replay.extend(result["transitions"])
            bucket = per_env_stats.setdefault(
                result["env_key"],
                {
                    "rewards": [],
                    "reward_per_fe": [],
                    "init_hv": [],
                    "final_hv": [],
                },
            )
            ep_reward = float(result["episode_reward"])
            ep_steps = max(int(result.get("episode_steps", 0)), 1)
            bucket["rewards"].append(ep_reward)
            bucket["reward_per_fe"].append(ep_reward / float(ep_steps))
            bucket["init_hv"].append(float(result["init_hv"]))
            bucket["final_hv"].append(float(result["final_hv"]))
        per_env_summaries = {}
        for key, stats in sorted(per_env_stats.items()):
            if len(stats["rewards"]) == 0:
                continue
            per_env_summaries[key] = {
                "mean_reward": float(np.mean(stats["rewards"])),
                "mean_reward_per_fe": float(np.mean(stats["reward_per_fe"])),
                "init_hv": float(np.mean(stats["init_hv"])),
                "final_hv": float(np.mean(stats["final_hv"])),
            }
        mean_ep_reward = (
            float(np.mean([v["mean_reward_per_fe"] for v in per_env_summaries.values()]))
            if per_env_summaries
            else 0.0
        )
        for key, stats in per_env_summaries.items():
            log(
                f"{key} epoch {epoch} done, "
                f"mean reward/FE = {stats['mean_reward_per_fe']:.4f}, "
                f"init HV = {stats['init_hv']:.6f}, "
                f"final HV = {stats['final_hv']:.6f}"
            )
        update_start_time = time.perf_counter()

        if len(replay) < cfg.batch_size:
            empty_metrics = {
                "q_mean": float("nan"),
                "q_std": float("nan"),
                "target_mean": float("nan"),
                "td_error_mean": float("nan"),
                "td_error_std": float("nan"),
                "reward_mean": float("nan"),
                "shape_group": 0,
                "group_sizes": [],
                "shape_group_summary": {},
            }
            log(
                f"[Epoch {epoch:04d}] "
                f"set={cfg.training_set} | heldout={cfg.heldout_problem} | "
                f"envs_active={len(env_specs)}/{len(env_specs)} | "
                f"replay={len(replay)} | skip update | ep_return_per_fe={mean_ep_reward:.4f}"
            )
            log(
                f"epoch {epoch} done | mean reward/FE = {mean_ep_reward:.4f} | "
                f"set = {cfg.training_set} | heldout = {cfg.heldout_problem} | "
            f"surrogate = {cfg.surrogate_model} | sur_steps = {cfg.surrogate_nsga_steps} | "
                f"workers = {actual_num_workers} | replay = {len(replay)} | "
                f"reward_scheme = {cfg.reward_scheme} | policy = {cfg.policy_mode} | update = skipped"
                f" | update_time_sec = {time.perf_counter() - update_start_time:.3f}"
                f" | td_loss = nan | grad_norm = nan | q_mean = {empty_metrics['q_mean']} | "
                f"q_std = {empty_metrics['q_std']} | target_mean = {empty_metrics['target_mean']} | "
                f"td_error_mean = {empty_metrics['td_error_mean']} | "
                f"td_error_std = {empty_metrics['td_error_std']} | "
                f"shape_group = {empty_metrics['shape_group']} | "
                f"group_sizes = {empty_metrics['shape_group_summary']}"
            )
            if mean_ep_reward > best_reward:
                log(f"new best mean reward at epoch {epoch}: {mean_ep_reward:.4f}")
            best_reward = save_training_checkpoint(
                agent,
                cfg,
                cfg.heldout_problem,
                epoch,
                mean_ep_reward,
                best_reward,
                best_state_dict=pre_update_state_dict,
            )
            continue

        update_metrics_list = []
        agent.train()
        for update_idx in range(int(cfg.updates_per_epoch)):
            batch = replay.sample(cfg.batch_size)
            loss, ddqn_metrics = compute_ddqn_loss(agent, target_agent, batch, cfg)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0))
            optimizer.step()

            update_metrics = {
                "td_loss": float(loss.item()),
                "grad_norm": float(grad_norm),
                "q_mean": float(ddqn_metrics["q_mean"]),
                "q_std": float(ddqn_metrics["q_std"]),
                "target_mean": float(ddqn_metrics["target_mean"]),
                "td_error_mean": float(ddqn_metrics["td_error_mean"]),
                "td_error_std": float(ddqn_metrics["td_error_std"]),
                "reward_mean": float(ddqn_metrics["reward_mean"]),
                "shape_group": int(ddqn_metrics["shape_group"]),
                "group_sizes": list(ddqn_metrics["group_sizes"]),
                "shape_group_detail": dict(ddqn_metrics.get("shape_group_detail", {})),
            }
            update_metrics_list.append(update_metrics)

        if it % cfg.target_update_interval == 0:
            target_agent.load_state_dict(agent.state_dict())
        update_elapsed = time.perf_counter() - update_start_time

        mean_update_metrics = {}
        for key in ["td_loss", "grad_norm", "q_mean", "q_std", "target_mean", "td_error_mean", "td_error_std", "reward_mean"]:
            mean_update_metrics[key] = float(np.mean([m[key] for m in update_metrics_list]))
        mean_update_metrics["shape_group"] = int(round(np.mean([m["shape_group"] for m in update_metrics_list])))
        group_keys = sorted(
            {
                int(obj)
                for m in update_metrics_list
                for obj in m.get("shape_group_detail", {}).keys()
            }
        )
        mean_update_metrics["shape_group_summary"] = {
            int(obj): float(np.mean([m.get("shape_group_detail", {}).get(int(obj), 0) for m in update_metrics_list]))
            for obj in group_keys
        }

        log(
            f"[Epoch {epoch:04d}] "
            f"set={cfg.training_set} | "
            f"heldout={cfg.heldout_problem} | "
            f"envs_active={len(env_specs)}/{len(env_specs)} | "
            f"surrogate={cfg.surrogate_model} | "
            f"sur_steps={cfg.surrogate_nsga_steps} | "
            f"updates={cfg.updates_per_epoch} | "
            f"update_time_sec={update_elapsed:.3f} | "
            f"eps={epsilon:.3f} | "
            f"replay={len(replay)} | "
            f"td_loss={mean_update_metrics['td_loss']:.6f} | "
            f"grad_norm={mean_update_metrics['grad_norm']:.6f} | "
            f"q_mean={mean_update_metrics['q_mean']:.4f} | "
            f"q_std={mean_update_metrics['q_std']:.4f} | "
            f"target_mean={mean_update_metrics['target_mean']:.4f} | "
            f"td_error_mean={mean_update_metrics['td_error_mean']:.4f} | "
            f"td_error_std={mean_update_metrics['td_error_std']:.4f} | "
            f"shape_group={mean_update_metrics['shape_group']} | "
            f"group_sizes={mean_update_metrics['shape_group_summary']} | "
            f"batch_r={mean_update_metrics['reward_mean']:.4f} | "
            f"ep_return_per_fe={mean_ep_reward:.4f}"
        )
        log(
            f"epoch {epoch} done | mean reward/FE = {mean_ep_reward:.4f} | "
            f"set = {cfg.training_set} | heldout = {cfg.heldout_problem} | "
            f"surrogate = {cfg.surrogate_model} | sur_steps = {cfg.surrogate_nsga_steps} | "
            f"workers = {actual_num_workers} | replay = {len(replay)} | "
            f"reward_scheme = {cfg.reward_scheme} | policy = {cfg.policy_mode} | "
            f"updates = {len(update_metrics_list)} | "
            f"update_time_sec = {update_elapsed:.3f} | "
            f"td_loss = {mean_update_metrics['td_loss']:.6f} | grad_norm = {mean_update_metrics['grad_norm']:.6f} | "
            f"q_mean = {mean_update_metrics['q_mean']:.4f} | q_std = {mean_update_metrics['q_std']:.4f} | "
            f"target_mean = {mean_update_metrics['target_mean']:.4f} | "
            f"td_error_mean = {mean_update_metrics['td_error_mean']:.4f} | "
            f"td_error_std = {mean_update_metrics['td_error_std']:.4f} | "
            f"shape_group = {mean_update_metrics['shape_group']} | "
            f"group_sizes = {mean_update_metrics['shape_group_summary']}"
        )

        if mean_ep_reward > best_reward:
            log(f"new best mean reward at epoch {epoch}: {mean_ep_reward:.4f}")
        best_reward = save_training_checkpoint(
            agent,
            cfg,
            cfg.heldout_problem,
            epoch,
            mean_ep_reward,
            best_reward,
            best_state_dict=pre_update_state_dict,
        )

    if executor is not None:
        executor.shutdown(wait=True)
    if bool(use_ray) and ray_mod is not None and ray_mod.is_initialized():
        ray_mod.shutdown()
    log_fp.close()
    return agent


if __name__ == "__main__":
    args = parse_args()
    train_disc_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        epoch=args.epoch,
        reward_scheme=int(args.reward_scheme),
        surrogate_model=str(args.surrogate_model),
        training_set=int(args.training_set),
        num_workers=args.num_workers,
        surrogate_nsga_steps=int(args.surrogate_nsga_steps),
        updates_per_epoch=args.updates_per_epoch,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        use_ray=bool(args.ray),
    )
