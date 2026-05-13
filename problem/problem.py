from __future__ import annotations

import numpy as np

SUPPORTED_PROBLEMS: list[str] = [
    "ZDT1",
    "ZDT2",
    "ZDT3",
    "ZDT7",
    "DTLZ1",
    "DTLZ2",
    "DTLZ3",
    "DTLZ4",
    "DTLZ5",
    "DTLZ6",
    "DTLZ7",
]


class ZDTProblem:
    """ZDT benchmark problems (minimization)."""

    def __init__(self, name: str, dim: int = 30):
        self.name = str(name).upper()
        self.dim = int(dim)
        self.lower = 0.0
        self.upper = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")

        f1 = x[:, 0]
        g = 1.0 + 9.0 / (self.dim - 1.0) * np.sum(x[:, 1:], axis=1)

        if self.name == "ZDT1":
            h = 1.0 - np.sqrt(f1 / g)
        elif self.name == "ZDT2":
            h = 1.0 - (f1 / g) ** 2
        elif self.name == "ZDT3":
            h = 1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1)
        elif self.name == "ZDT7":
            g = 1.0 + 10.0 * np.sum(x[:, 1:], axis=1) / (self.dim - 1.0)
            h = 1.0 - (f1 / g) * (1.0 + np.sin(3.0 * np.pi * f1))
        else:
            raise ValueError(f"Unsupported problem: {self.name}")

        f2 = g * h
        return np.stack([f1, f2], axis=1).astype(np.float32)


class DTLZProblem:
    """DTLZ benchmark problems (minimization)."""

    def __init__(self, name: str, dim: int = 30):
        self.name = str(name).upper()
        self.dim = int(dim)
        self.lower = 0.0
        self.upper = 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={x.shape}")

        if self.name == "DTLZ1":
            g = 100.0 * (
                self.dim - 1.0
                + np.sum((x[:, 1:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, 1:] - 0.5)), axis=1)
            )
            f1 = 0.5 * x[:, 0] * (1.0 + g)
            f2 = 0.5 * (1.0 - x[:, 0]) * (1.0 + g)
            return np.stack([f1, f2], axis=1).astype(np.float32)

        if self.name == "DTLZ2":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = 0.5 * np.pi * x[:, 1]
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ3":
            g = 100.0 * (
                self.dim - 2.0
                + np.sum((x[:, 2:] - 0.5) ** 2 - np.cos(20.0 * np.pi * (x[:, 2:] - 0.5)), axis=1)
            )
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = 0.5 * np.pi * x[:, 1]
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ4":
            alpha = 100.0
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * (x[:, 0] ** alpha)
            theta2 = 0.5 * np.pi * (x[:, 1] ** alpha)
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ5":
            g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, 1])
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ6":
            g = np.sum(x[:, 2:] ** 0.1, axis=1)
            theta1 = 0.5 * np.pi * x[:, 0]
            theta2 = (np.pi / (4.0 * (1.0 + g))) * (1.0 + 2.0 * g * x[:, 1])
            f1 = (1.0 + g) * np.cos(theta1) * np.cos(theta2)
            f2 = (1.0 + g) * np.cos(theta1) * np.sin(theta2)
            f3 = (1.0 + g) * np.sin(theta1)
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        if self.name == "DTLZ7":
            denom = max(float(self.dim) - 2.0, 1.0)
            g = 1.0 + 9.0 * np.sum(x[:, 2:], axis=1) / denom
            f1 = x[:, 0]
            f2 = x[:, 1]
            term1 = (f1 / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * f1))
            term2 = (f2 / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * f2))
            h = 3.0 - (term1 + term2)
            f3 = (1.0 + g) * h
            y = np.stack([f1, f2, f3], axis=1).astype(np.float32)
            return np.maximum(y, 0.0)

        raise ValueError(f"Unsupported problem: {self.name}")


def make_problem(name: str, dim: int = 30):
    key = str(name).upper()
    if key.startswith("ZDT"):
        return ZDTProblem(key, dim=int(dim))
    if key.startswith("DTLZ"):
        return DTLZProblem(key, dim=int(dim))
    raise ValueError(f"Unsupported problem: {name}")
