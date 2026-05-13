import os
import argparse
import copy
import random
from dataclasses import dataclass
from collections import deque

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.disc import Disc
from nsga2_solver import run_surrogate_nsga2
from problem.problem import make_problem
from ref_points_hv import get_reference_point
from reward import pareto_front, reward_scheme_1, reward_scheme_2, reward_scheme_3
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
    episodes_per_worker: int = 2
    max_fe: int = 120
    init_size: int = 80
    batch_size: int = 128
    replay_size: int = 50000
    gamma: float = 1.0
    lr: float = 1e-4
    target_update_interval: int = 20
    train_iters: int = 2000
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    epsilon_decay_iters: int = 1000
    hidden_dim: int = 128
    n_heads: int = 8
    ff_dim: int = 256
    dropout: float = 0.0
    logit_scale: float = 1.0
    surrogate_model: str = "kan"
    surrogate_nsga_steps: int = 100
    offspring_size: int = 80
    kan_steps: int = 100
    kan_hidden_width: int = 64
    kan_grid: int = 5
    reward_scheme: int = 1
    training_set: int = 1
    heldout_problem: str = "ZDT1"
    weight_dir: str = "weight"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--reward_scheme", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--surrogate_model", type=str, default="kan", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--training_set", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
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
            device=str(cfg_dict.get("device", "cpu")),
            kan_steps=int(cfg_dict.get("kan_steps", 100)),
            hidden_width=int(cfg_dict.get("kan_hidden_width", 64)),
            grid=int(cfg_dict.get("kan_grid", 5)),
            seed=int(cfg_dict.get("seed", 0)),
        )
        return KANSurrogateModel(models=models, device=str(cfg_dict.get("device", "cpu")))

    if surrogate_name == "tabpfn":
        return fit_tabpfn_surrogate(
            archive_x=np.asarray(archive_x, dtype=np.float32),
            archive_y=np.asarray(archive_y, dtype=np.float32),
            device=str(cfg_dict.get("device", "cpu")),
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


def pad_stack_rows(arrays, pad_value=0.0, mode="repeat_last"):
    arrays = [np.asarray(arr, dtype=np.float32) for arr in arrays]
    max_rows = max(int(arr.shape[0]) for arr in arrays)
    padded = []
    for arr in arrays:
        n_rows = int(arr.shape[0])
        if n_rows == max_rows:
            padded.append(arr)
            continue
        n_pad = max_rows - n_rows
        if mode == "repeat_last" and n_rows > 0:
            pad = np.repeat(arr[-1:, :], n_pad, axis=0)
        else:
            pad_shape = (n_pad,) + arr.shape[1:]
            pad = np.full(pad_shape, pad_value, dtype=np.float32)
        padded.append(np.concatenate([arr, pad], axis=0))
    return np.stack(padded, axis=0)


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

    device = cfg.device

    x_true = to_tensor(pad_stack_rows(x_true), device)
    y_true = to_tensor(pad_stack_rows(y_true), device)
    x_sur = to_tensor(pad_stack_rows(x_sur), device)
    y_sur = to_tensor(pad_stack_rows(y_sur), device)
    sigma_sur = to_tensor(pad_stack_rows(sigma_sur), device)
    progress = to_tensor(np.asarray(progress).reshape(-1, 1), device)

    lower_bound = to_tensor(np.stack(lower_bound), device)
    upper_bound = to_tensor(np.stack(upper_bound), device)

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
            decode_type="q_greedy",
        )

        next_q = next_target["q_values"].gather(1, next_actions.view(-1, 1)).squeeze(1)
        target = rewards + cfg.gamma * next_q * (1.0 - dones)

    loss = nn.SmoothL1Loss()(q_sa, target)
    return loss, q_sa.detach().mean().item(), rewards.mean().item()


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

    def _progress(self):
        return float(self.t) / float(max(self.max_steps - 1, 1))

    def _surrogate_cfg(self):
        cfg_local = dict(self.cfg)
        cfg_local["seed"] = int(self.seed) + int(self.t)
        return cfg_local

    def _fit_surrogate(self):
        return build_surrogate_from_cfg(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
        )

    def _refresh_offspring(self):
        surrogate = self._fit_surrogate()
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
        self.nsga2_problem = make_nsga2_problem_adapter(self.problem, int(self.archive_y.shape[1]))
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
        self._refresh_offspring()
        return self._build_state(), float(reward), bool(done)


@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, worker_id, cfg_dict, env_specs):
        self.worker_id = worker_id
        self.cfg = cfg_dict
        self.env_specs = list(env_specs)

    def rollout(self, state_dict_cpu, epsilon, episodes):
        device = "cpu"

        agent = Disc(
            hidden_dim=self.cfg["hidden_dim"],
            n_heads=self.cfg["n_heads"],
            ff_dim=self.cfg["ff_dim"],
            dropout=self.cfg["dropout"],
            logit_scale=self.cfg["logit_scale"],
            epsilon=epsilon,
        ).to(device)

        agent.load_state_dict(state_dict_cpu)
        agent.eval()

        transitions = []
        ep_rewards = []
        env_rewards = {}
        for env_idx, spec in enumerate(self.env_specs):
            key = env_key(spec["problem_name"], spec["dim"])
            env_rewards[key] = []

            for ep in range(episodes):
                env = DiscSAEAEnv(
                    problem_name=spec["problem_name"],
                    dim=int(spec["dim"]),
                    seed=100000 * self.worker_id + 1000 * env_idx + ep,
                    cfg_dict=self.cfg,
                )
                state = env.reset()
                total_reward = 0.0

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
                            decode_type="epsilon_greedy",
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

                ep_rewards.append(total_reward)
                env_rewards[key].append(float(total_reward))

        return transitions, ep_rewards, env_rewards


def save_training_checkpoint(agent, cfg, problem_name, epoch, mean_reward, best_reward):
    os.makedirs(cfg.weight_dir, exist_ok=True)
    rs_tag = f"rs{int(cfg.reward_scheme)}"
    problem_tag = str(problem_name).lower()

    if mean_reward > best_reward:
        best_path = os.path.join(cfg.weight_dir, f"disc_problem_{problem_tag}_{rs_tag}_best_reward.pth")
        torch.save(
            {
                "epoch": int(epoch),
                "problem_name": str(problem_name),
                "reward_scheme": int(cfg.reward_scheme),
                "mean_reward": float(mean_reward),
                "state_dict": agent.state_dict(),
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
    reward_scheme=1,
    surrogate_model="kan",
    training_set=1,
    num_workers=None,
    surrogate_nsga_steps=100,
):
    cfg = TrainConfig()
    cfg.reward_scheme = int(reward_scheme)
    cfg.surrogate_model = str(surrogate_model).lower()
    cfg.training_set = int(training_set)
    cfg.heldout_problem = str(problem_name).upper()
    cfg.surrogate_nsga_steps = int(surrogate_nsga_steps)
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
    actual_num_workers = min(int(cfg.num_workers), len(env_specs))
    cfg_dict = cfg.__dict__.copy()

    if not ray.is_initialized():
        ray.init(num_cpus=actual_num_workers, ignore_reinit_error=True)

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
    replay = ReplayBuffer(cfg.replay_size)
    best_reward = -float("inf")

    worker_env_specs = [env_specs[i::actual_num_workers] for i in range(actual_num_workers)]
    workers = [
        RolloutWorker.remote(i, cfg_dict, worker_env_specs[i])
        for i in range(actual_num_workers)
        if len(worker_env_specs[i]) > 0
    ]

    for it in range(cfg.train_iters):
        epoch = int(it) + 1
        epsilon = epsilon_by_iter(it, cfg)

        print(
            f"[Epoch {epoch:04d}] start | "
            f"set={cfg.training_set} | "
            f"heldout={cfg.heldout_problem} | "
            f"envs_active={len(env_specs)}/{len(env_specs)} | "
            f"surrogate={cfg.surrogate_model} | "
            f"sur_steps={cfg.surrogate_nsga_steps} | "
            f"eps={epsilon:.3f}"
        )

        state_cpu = clone_state_dict_cpu(agent)
        futures = [
            w.rollout.remote(
                state_dict_cpu=state_cpu,
                epsilon=epsilon,
                episodes=cfg.episodes_per_worker,
            )
            for w in workers
        ]
        results = ray.get(futures)

        all_rewards = []
        per_env_reward_lists = {}
        for transitions, ep_rewards, env_rewards in results:
            replay.extend(transitions)
            all_rewards.extend(ep_rewards)
            for key, values in env_rewards.items():
                per_env_reward_lists.setdefault(key, []).extend(float(v) for v in values)

        mean_ep_reward = float(np.mean(all_rewards)) if all_rewards else 0.0
        per_env_mean_rewards = {
            key: float(np.mean(values))
            for key, values in sorted(per_env_reward_lists.items())
            if len(values) > 0
        }
        env_reward_log = " | ".join(
            f"{key}={value:.4f}" for key, value in per_env_mean_rewards.items()
        )

        if len(replay) < cfg.batch_size:
            print(
                f"[Epoch {epoch:04d}] "
                f"set={cfg.training_set} | heldout={cfg.heldout_problem} | "
                f"envs_active={len(env_specs)}/{len(env_specs)} | "
                f"replay={len(replay)} | skip update | ep_return={mean_ep_reward:.4f}"
            )
            if env_reward_log:
                print(f"[Epoch {epoch:04d}] env_mean_reward | {env_reward_log}")
            best_reward = save_training_checkpoint(agent, cfg, cfg.heldout_problem, epoch, mean_ep_reward, best_reward)
            continue

        batch = replay.sample(cfg.batch_size)

        agent.train()
        loss, q_mean, r_mean = compute_ddqn_loss(agent, target_agent, batch, cfg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        optimizer.step()

        if it % cfg.target_update_interval == 0:
            target_agent.load_state_dict(agent.state_dict())

        print(
            f"[Epoch {epoch:04d}] "
            f"set={cfg.training_set} | "
            f"heldout={cfg.heldout_problem} | "
            f"envs_active={len(env_specs)}/{len(env_specs)} | "
            f"surrogate={cfg.surrogate_model} | "
            f"sur_steps={cfg.surrogate_nsga_steps} | "
            f"eps={epsilon:.3f} | "
            f"replay={len(replay)} | "
            f"loss={loss.item():.6f} | "
            f"q_mean={q_mean:.4f} | "
            f"batch_r={r_mean:.4f} | "
            f"ep_return={mean_ep_reward:.4f}"
        )
        if env_reward_log:
            print(f"[Epoch {epoch:04d}] env_mean_reward | {env_reward_log}")

        best_reward = save_training_checkpoint(agent, cfg, cfg.heldout_problem, epoch, mean_ep_reward, best_reward)

    return agent


if __name__ == "__main__":
    args = parse_args()
    train_disc_ddqn_ray(
        problem_name=args.problem,
        dim=int(args.dim),
        reward_scheme=int(args.reward_scheme),
        surrogate_model=str(args.surrogate_model),
        training_set=int(args.training_set),
        num_workers=args.num_workers,
        surrogate_nsga_steps=int(args.surrogate_nsga_steps),
    )
