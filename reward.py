from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance for minimization."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return bool(np.all(a <= b) and np.any(a < b))


def pareto_front(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.reshape(0, 0).astype(np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must be 2D, got shape={values.shape}")

    keep: list[int] = []
    for i in range(values.shape[0]):
        dominated = False
        for j in range(values.shape[0]):
            if i == j:
                continue
            if _dominates(values[j], values[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return values[np.asarray(keep, dtype=np.int64)]


def hypervolume(values: np.ndarray, ref_point: np.ndarray) -> float:
    front = pareto_front(values)
    if front.size == 0:
        return 0.0
    return float(HV(ref_point=np.asarray(ref_point, dtype=np.float32))(front))


def hv_improvement_reward(
    *,
    previous_archive: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    epsilon: float = 1e-8,
    no_improve_reward: float = -1.0,
) -> float:
    previous_archive = np.asarray(previous_archive, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined = np.vstack([previous_archive, selected_objectives])

    prev_hv = hypervolume(previous_archive, ref_point)
    next_hv = hypervolume(combined, ref_point)
    if next_hv <= prev_hv:
        return float(no_improve_reward)
    return float((next_hv - prev_hv) / (prev_hv + float(epsilon)))


def fpareto_improvement_reward(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    no_improve_reward: float = -1.0,
) -> float:
    """Legacy 'fpareto' reward used by older demos (distance-to-front with an improvement gate)."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)

    improved = False
    for candidate in selected_objectives:
        if not any(_dominates(prev, candidate) for prev in previous_front):
            improved = True
            break
    if not improved:
        return float(no_improve_reward)

    reward = 1.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in selected_objectives:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(reward)


def reward_scheme_1(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Distance-to-front reward, scaled and offset; returns 0 if HV doesn't improve."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(np.vstack([previous_front, selected_objectives]), ref_point)
    if next_hv <= prev_hv:
        return -1.0

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in selected_objectives:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(max(1e-6, 1.0 + 10.0 * reward))


def reward_scheme_2(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
) -> float:
    """Distance-to-front reward; returns 0 if HV doesn't improve."""
    previous_front = pareto_front(np.asarray(previous_front, dtype=np.float32))
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(np.vstack([previous_front, selected_objectives]), ref_point)
    if next_hv <= prev_hv:
        return 0.0

    reward = 0.0
    origin = np.zeros(previous_front.shape[1], dtype=np.float32)
    for candidate in selected_objectives:
        distances = np.abs(previous_front - candidate).sum(axis=1)
        nearest_idx = int(np.argmin(distances))
        d_i = float(distances[nearest_idx])
        d_ref_i = float(np.abs(previous_front[nearest_idx] - origin).sum())
        reward += d_i / max(d_ref_i, 1e-12)
    return float(max(1e-6, 10.0 * reward))


def reward_scheme_3(
    *,
    previous_front: np.ndarray,
    selected_objectives: np.ndarray,
    ref_point: np.ndarray,
    hv_epsilon: float = 1e-8,
) -> float:
    """Scaled normalized HV improvement; returns 0 if HV doesn't improve."""
    previous_front = np.asarray(previous_front, dtype=np.float32)
    selected_objectives = np.asarray(selected_objectives, dtype=np.float32)
    combined_front = np.vstack([previous_front, selected_objectives])

    prev_hv = hypervolume(previous_front, ref_point)
    next_hv = hypervolume(combined_front, ref_point)
    if next_hv <= prev_hv:
        return 0.0
    return float(50.0 * (next_hv - prev_hv) / (prev_hv + float(hv_epsilon)))
