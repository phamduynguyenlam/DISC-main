from __future__ import annotations

import argparse
import copy
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from agents.disc import Disc
from trainer import (
    DiscSAEAEnv,
    ReplayBuffer,
    TrainConfig,
    build_surrogate_from_cfg,
    build_training_env_specs,
    clone_state_dict_cpu,
    compute_ddqn_loss,
    env_key,
    epsilon_by_iter,
    rollout_episode_task_local,
    save_training_checkpoint,
    select_action_from_output,
    to_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DISC with batched TabPFN surrogate-assisted environments.")
    parser.add_argument("--problem", type=str, default="ZDT1")
    parser.add_argument("--dim", type=int, default=30)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--reward_scheme", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--surrogate_model", type=str, default="tabpfn", choices=["gp", "kan", "tabpfn"])
    parser.add_argument("--training_set", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--surrogate_nsga_steps", type=int, default=100)
    parser.add_argument("--updates_per_epoch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--rollout_device", type=str, default="cpu")
    parser.add_argument("--surrogate_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ray", action="store_true")
    return parser.parse_args()


def _is_cuda_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def _use_tabpfn_gpu_batch(cfg_dict: dict[str, Any]) -> bool:
    surrogate_name = str(cfg_dict.get("surrogate_model", "gp")).lower()
    surrogate_device = str(cfg_dict.get("surrogate_device", "cpu"))
    return surrogate_name == "tabpfn" and _is_cuda_device(surrogate_device)


@dataclass
class _TabPFNBatchRequest:
    backend: Any
    mode: str
    x: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: BaseException | None = None


class TabPFNBatchCoordinator:
    def __init__(self, expected_requesters: int, wait_timeout_s: float = 0.002):
        self.expected_requesters = max(1, int(expected_requesters))
        self.wait_timeout_s = max(0.0, float(wait_timeout_s))
        self._pending: list[_TabPFNBatchRequest] = []
        self._closed = False
        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._serve, name="tabpfn-batch-worker", daemon=True)
        self._worker.start()

    def infer(self, backend: Any, mode: str, x: np.ndarray) -> np.ndarray:
        req = _TabPFNBatchRequest(
            backend=backend,
            mode=str(mode),
            x=np.asarray(x, dtype=np.float32).copy(),
        )
        with self._cv:
            if self._closed:
                raise RuntimeError("TabPFNBatchCoordinator is already closed.")
            self._pending.append(req)
            self._cv.notify_all()
        req.done.wait()
        if req.error is not None:
            raise req.error
        if req.result is None:
            raise RuntimeError("TabPFN batch request completed without a result.")
        return np.asarray(req.result, dtype=np.float32)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._worker.join()

    def _serve(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                if self._closed and not self._pending:
                    return
                if len(self._pending) < self.expected_requesters and not self._closed:
                    self._cv.wait(timeout=self.wait_timeout_s)
                batch = self._pending
                self._pending = []

            groups: dict[tuple[int, str], list[_TabPFNBatchRequest]] = {}
            for req in batch:
                key = (id(req.backend), req.mode)
                groups.setdefault(key, []).append(req)

            for (_, mode), requests in groups.items():
                backend = requests[0].backend
                x_parts = [np.asarray(req.x, dtype=np.float32) for req in requests]
                offsets = np.cumsum([0] + [int(part.shape[0]) for part in x_parts])
                x_batch = np.concatenate(x_parts, axis=0).astype(np.float32)
                try:
                    if mode == "mean":
                        pred_batch = _tabpfn_predict_mean_backend(backend, x_batch)
                    elif mode == "std":
                        pred_batch = _tabpfn_predict_std_backend(backend, x_batch)
                    else:
                        raise ValueError(f"Unsupported TabPFN batch mode: {mode}")
                except BaseException as exc:
                    for req in requests:
                        req.error = exc
                        req.done.set()
                    continue

                for req_idx, req in enumerate(requests):
                    lo = int(offsets[req_idx])
                    hi = int(offsets[req_idx + 1])
                    req.result = np.asarray(pred_batch[lo:hi], dtype=np.float32)
                    req.done.set()


def _tabpfn_predict_mean_backend(backend: Any, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(backend, "predict_mean"):
        return np.asarray(backend.predict_mean(x_arr), dtype=np.float32)
    if hasattr(backend, "predict"):
        return np.asarray(backend.predict(x_arr), dtype=np.float32)
    if hasattr(backend, "predict_mean_std"):
        mean, _ = backend.predict_mean_std(x_arr)
        return np.asarray(mean, dtype=np.float32)
    raise AttributeError(f"{type(backend).__name__} does not implement predict_mean(), predict(), or predict_mean_std().")


def _tabpfn_predict_std_backend(backend: Any, x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float32)
    if hasattr(backend, "predict_std"):
        return np.asarray(backend.predict_std(x_arr), dtype=np.float32)
    if hasattr(backend, "predict_mean_std"):
        _, std = backend.predict_mean_std(x_arr)
        return np.asarray(std, dtype=np.float32)
    raise AttributeError(f"{type(backend).__name__} does not implement predict_std() or predict_mean_std().")


class BatchedTabPFNSurrogate:
    def __init__(self, backend: Any, coordinator: TabPFNBatchCoordinator):
        self.backend = backend
        self.coordinator = coordinator

    def predict_mean(self, x: np.ndarray, device: str | None = None) -> np.ndarray:
        del device
        return self.coordinator.infer(self.backend, "mean", np.asarray(x, dtype=np.float32))

    def predict_std(self, x: np.ndarray) -> np.ndarray:
        return self.coordinator.infer(self.backend, "std", np.asarray(x, dtype=np.float32))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_mean(x)


class TabPFNDiscSAEAEnv(DiscSAEAEnv):
    def __init__(self, problem_name, dim, seed, cfg_dict, tabpfn_batcher: TabPFNBatchCoordinator | None = None):
        super().__init__(problem_name=problem_name, dim=dim, seed=seed, cfg_dict=cfg_dict)
        self.tabpfn_batcher = tabpfn_batcher

    def _fit_surrogate(self):
        surrogate = build_surrogate_from_cfg(
            self._surrogate_cfg(),
            archive_x=self.archive_x,
            archive_y=self.archive_y,
        )
        if self.tabpfn_batcher is not None and _use_tabpfn_gpu_batch(self.cfg):
            surrogate = BatchedTabPFNSurrogate(surrogate, self.tabpfn_batcher)
        self.surrogate = surrogate
        return self.surrogate


def rollout_episode_task_batched(
    state_dict_cpu,
    cfg_dict,
    problem_name,
    dim,
    seed,
    epsilon,
    tabpfn_batcher: TabPFNBatchCoordinator | None = None,
):
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

    env = TabPFNDiscSAEAEnv(
        problem_name=problem_name,
        dim=int(dim),
        seed=int(seed),
        cfg_dict=cfg_dict,
        tabpfn_batcher=tabpfn_batcher,
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
        transitions.append(
            (
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
            )
        )
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


def train_disc_ddqn_tabpfn(
    problem_name="ZDT1",
    dim=30,
    epoch=None,
    reward_scheme=1,
    surrogate_model="tabpfn",
    training_set=1,
    surrogate_nsga_steps=100,
    updates_per_epoch=None,
    device=None,
    rollout_device="cpu",
    surrogate_device="cuda",
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
    if cfg.surrogate_model not in {"gp", "kan", "tabpfn"}:
        raise ValueError(
            f"Unsupported surrogate_model: {cfg.surrogate_model}. "
            "Expected one of {'gp', 'kan', 'tabpfn'}."
        )

    env_specs = build_training_env_specs(cfg.heldout_problem, cfg.training_set)
    cfg.num_workers = int(len(env_specs))
    if int(cfg.train_iters) <= 0:
        raise ValueError(f"epoch must be positive, got {cfg.train_iters}.")
    actual_num_workers = int(len(env_specs))
    cfg_dict = cfg.__dict__.copy()
    use_batched_tabpfn = _use_tabpfn_gpu_batch(cfg_dict)
    if bool(use_ray) and use_batched_tabpfn:
        raise ValueError("tabpfn_trainer.py does not support --ray together with surrogate_model=tabpfn and surrogate_device=cuda.")

    os.makedirs("training_logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        "training_logs",
        f"tabpfn_trainer_{cfg.heldout_problem.lower()}_set{cfg.training_set}_{ts}.txt",
    )
    log_fp = open(log_path, "a", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg)
        log_fp.write(str(msg) + "\n")

    executor = None
    ray_rollout_remote = None
    ray_mod = None
    tabpfn_batcher = None
    sampling_backend = "process_pool"

    if use_batched_tabpfn:
        sampling_backend = "tabpfn_gpu_batch"
        tabpfn_batcher = TabPFNBatchCoordinator(expected_requesters=actual_num_workers)
        executor = ThreadPoolExecutor(max_workers=actual_num_workers)
    elif bool(use_ray):
        sampling_backend = "ray"
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
        f"sampling_backend={sampling_backend} | "
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

    try:
        for it in range(cfg.train_iters):
            epoch_id = int(it) + 1
            epsilon = epsilon_by_iter(it, cfg)
            replay = ReplayBuffer(cfg.replay_size)

            log(
                f"[Epoch {epoch_id:04d}] start | "
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
                    seed = 100000 * epoch_id + 1000 * env_idx + ep
                    if use_batched_tabpfn:
                        futures.append(
                            executor.submit(
                                rollout_episode_task_batched,
                                state_cpu,
                                cfg_dict,
                                spec["problem_name"],
                                int(spec["dim"]),
                                int(seed),
                                epsilon,
                                tabpfn_batcher,
                            )
                        )
                    elif bool(use_ray):
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
                    f"{key} epoch {epoch_id} done, "
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
                    "shape_group_summary": {},
                }
                log(
                    f"[Epoch {epoch_id:04d}] "
                    f"set={cfg.training_set} | heldout={cfg.heldout_problem} | "
                    f"envs_active={len(env_specs)}/{len(env_specs)} | "
                    f"replay={len(replay)} | skip update | ep_return_per_fe={mean_ep_reward:.4f}"
                )
                log(
                    f"epoch {epoch_id} done | mean reward/FE = {mean_ep_reward:.4f} | "
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
                    log(f"new best mean reward at epoch {epoch_id}: {mean_ep_reward:.4f}")
                best_reward = save_training_checkpoint(
                    agent,
                    cfg,
                    cfg.heldout_problem,
                    epoch_id,
                    mean_ep_reward,
                    best_reward,
                    best_state_dict=pre_update_state_dict,
                )
                continue

            update_metrics_list = []
            agent.train()
            for _ in range(int(cfg.updates_per_epoch)):
                batch = replay.sample(cfg.batch_size)
                loss, ddqn_metrics = compute_ddqn_loss(agent, target_agent, batch, cfg)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0))
                optimizer.step()

                update_metrics_list.append(
                    {
                        "td_loss": float(loss.item()),
                        "grad_norm": float(grad_norm),
                        "q_mean": float(ddqn_metrics["q_mean"]),
                        "q_std": float(ddqn_metrics["q_std"]),
                        "target_mean": float(ddqn_metrics["target_mean"]),
                        "td_error_mean": float(ddqn_metrics["td_error_mean"]),
                        "td_error_std": float(ddqn_metrics["td_error_std"]),
                        "reward_mean": float(ddqn_metrics["reward_mean"]),
                        "shape_group": int(ddqn_metrics["shape_group"]),
                        "shape_group_detail": dict(ddqn_metrics.get("shape_group_detail", {})),
                    }
                )

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
                f"[Epoch {epoch_id:04d}] "
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
                f"epoch {epoch_id} done | mean reward/FE = {mean_ep_reward:.4f} | "
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
                log(f"new best mean reward at epoch {epoch_id}: {mean_ep_reward:.4f}")
            best_reward = save_training_checkpoint(
                agent,
                cfg,
                cfg.heldout_problem,
                epoch_id,
                mean_ep_reward,
                best_reward,
                best_state_dict=pre_update_state_dict,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        if tabpfn_batcher is not None:
            tabpfn_batcher.close()
        if bool(use_ray) and ray_mod is not None and ray_mod.is_initialized():
            ray_mod.shutdown()
        log_fp.close()

    return agent


if __name__ == "__main__":
    args = parse_args()
    train_disc_ddqn_tabpfn(
        problem_name=args.problem,
        dim=int(args.dim),
        epoch=args.epoch,
        reward_scheme=int(args.reward_scheme),
        surrogate_model=str(args.surrogate_model),
        training_set=int(args.training_set),
        surrogate_nsga_steps=int(args.surrogate_nsga_steps),
        updates_per_epoch=args.updates_per_epoch,
        device=args.device,
        rollout_device=str(args.rollout_device),
        surrogate_device=str(args.surrogate_device),
        use_ray=bool(args.ray),
    )
