from __future__ import annotations

import numpy as np


REFERENCE_POINTS: dict[str, np.ndarray] = {
    "ZDT1": np.array([0.9994, 6.0576], dtype=np.float32),
    "ZDT2": np.array([0.9994, 6.8960], dtype=np.float32),
    "ZDT3": np.array([0.9994, 6.0571], dtype=np.float32),
    "DTLZ2": np.array([2.8390, 2.9011, 2.8575], dtype=np.float32),
    "DTLZ3": np.array([2421.6427, 1905.2767, 2532.9691], dtype=np.float32),
    "DTLZ4": np.array([3.2675, 2.6443, 2.4263], dtype=np.float32),
    "DTLZ5": np.array([2.6672, 2.8009, 2.8575], dtype=np.float32),
    "DTLZ6": np.array([16.8258, 16.9194, 17.7646], dtype=np.float32),
    # DTLZ7 is treated as 3-objective (slice as needed when n_obj != 3).
    "DTLZ7": np.array([0.9984, 0.9961, 22.8114], dtype=np.float32),
    "RE1": np.array([2.76322289e03, 3.68876972e-02], dtype=np.float32),
    "RE2": np.array([528107.18990952, 1279320.81067113], dtype=np.float32),
    "RE3": np.array([7.68527849, 7.28609807, 21.50103909], dtype=np.float32),
    "RE4": np.array([6.79211111, 60.0, 0.4799612], dtype=np.float32),
    "RE5": np.array([0.87449713, 1.05091656, 1.05328528], dtype=np.float32),
    "RE6": np.array([749.92405125, 2229.37483405], dtype=np.float32),
    "RE7": np.array([2.10336300e02, 1.06991599e03, 3.91967702e07], dtype=np.float32),
}


def get_reference_point(problem_name: str, *, n_obj: int | None = None) -> np.ndarray:
    key = str(problem_name).upper()
    if key == "DTLZ1":
        if n_obj is None:
            n_obj = 2
        return np.full(int(n_obj), 1.1, dtype=np.float32)
    if key not in REFERENCE_POINTS:
        raise ValueError(f"Unsupported problem for reference point: {problem_name}")
    ref = np.asarray(REFERENCE_POINTS[key], dtype=np.float32).reshape(-1)
    if n_obj is None:
        return ref.astype(np.float32)
    return ref[: int(n_obj)].astype(np.float32)