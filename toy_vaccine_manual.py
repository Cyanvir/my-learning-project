"""
A tiny CPU-only demo of the Vaccine training idea.

This script intentionally uses only Python's standard library. It is not a
replacement for the real LLM code; it is a small model that demonstrates the
two-pass idea:

1. normal forward/backward to get gradient on the hidden representation
2. create a perturbation from that gradient
3. forward/backward again with hidden + perturbation, then update parameters

Run from the project root:

    python learning\\toy_vaccine_manual.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


@dataclass
class TinyModel:
    # h = W x + b, p = sigmoid(v dot h + c)
    W: list[list[float]]
    b: list[float]
    v: list[float]
    c: float

    @classmethod
    def init(cls, seed: int) -> "TinyModel":
        rng = random.Random(seed)
        return cls(
            W=[[rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)] for _ in range(2)],
            b=[0.0, 0.0],
            v=[rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2)],
            c=0.0,
        )

    def hidden(self, x: tuple[float, float]) -> list[float]:
        return [dot(row, list(x)) + bias for row, bias in zip(self.W, self.b)]

    def predict_proba(self, x: tuple[float, float], eps: list[float] | None = None) -> float:
        h = self.hidden(x)
        if eps is not None:
            h = [hi + ei for hi, ei in zip(h, eps)]
        return sigmoid(dot(self.v, h) + self.c)


def make_data(seed: int, n: int) -> list[tuple[tuple[float, float], int]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        x0 = rng.uniform(-1.5, 1.5)
        x1 = rng.uniform(-1.5, 1.5)
        score = x0 + 0.7 * x1 + 0.25 * math.sin(3 * x0)
        y = 1 if score > 0 else 0
        data.append(((x0, x1), y))
    return data


def zero_grads() -> dict[str, object]:
    return {
        "W": [[0.0, 0.0], [0.0, 0.0]],
        "b": [0.0, 0.0],
        "v": [0.0, 0.0],
        "c": 0.0,
    }


def add_sample_grads(
    model: TinyModel,
    grads: dict[str, object],
    x: tuple[float, float],
    y: int,
    eps: list[float] | None = None,
) -> float:
    h_clean = model.hidden(x)
    h = h_clean if eps is None else [hi + ei for hi, ei in zip(h_clean, eps)]
    p = sigmoid(dot(model.v, h) + model.c)
    dz = p - y

    grads["v"][0] += dz * h[0]  # type: ignore[index]
    grads["v"][1] += dz * h[1]  # type: ignore[index]
    grads["c"] += dz  # type: ignore[operator]

    # eps is treated as detached, just like the perturbation stored by hooks in
    # the real trainer. Gradients to W and b pass through h_clean.
    dh = [dz * model.v[0], dz * model.v[1]]
    for i in range(2):
        grads["b"][i] += dh[i]  # type: ignore[index]
        grads["W"][i][0] += dh[i] * x[0]  # type: ignore[index]
        grads["W"][i][1] += dh[i] * x[1]  # type: ignore[index]

    return -(y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12))


def apply_grads(model: TinyModel, grads: dict[str, object], lr: float, batch_size: int) -> None:
    scale = lr / batch_size
    for i in range(2):
        for j in range(2):
            model.W[i][j] -= scale * grads["W"][i][j]  # type: ignore[index]
        model.b[i] -= scale * grads["b"][i]  # type: ignore[index]
        model.v[i] -= scale * grads["v"][i]  # type: ignore[index]
    model.c -= scale * grads["c"]  # type: ignore[operator]


def hidden_grad(model: TinyModel, x: tuple[float, float], y: int) -> list[float]:
    p = model.predict_proba(x)
    dz = p - y
    return [dz * model.v[0], dz * model.v[1]]


def train_normal(model: TinyModel, data: list[tuple[tuple[float, float], int]], epochs: int, lr: float) -> None:
    for _ in range(epochs):
        random.shuffle(data)
        for start in range(0, len(data), 16):
            batch = data[start : start + 16]
            grads = zero_grads()
            for x, y in batch:
                add_sample_grads(model, grads, x, y)
            apply_grads(model, grads, lr, len(batch))


def train_vaccine_style(
    model: TinyModel,
    data: list[tuple[tuple[float, float], int]],
    epochs: int,
    lr: float,
    rho: float,
) -> None:
    for _ in range(epochs):
        random.shuffle(data)
        for start in range(0, len(data), 16):
            batch = data[start : start + 16]

            # First pass: collect gradients on hidden representations.
            first_pass_hidden_grads = [hidden_grad(model, x, y) for x, y in batch]
            flat = [g for pair in first_pass_hidden_grads for g in pair]
            grad_norm = norm(flat) + 1e-12
            eps_batch = [[rho * g / grad_norm for g in pair] for pair in first_pass_hidden_grads]

            # Second pass: train on perturbed hidden representations.
            grads = zero_grads()
            for (x, y), eps in zip(batch, eps_batch):
                add_sample_grads(model, grads, x, y, eps=eps)
            apply_grads(model, grads, lr, len(batch))


def accuracy(model: TinyModel, data: list[tuple[tuple[float, float], int]], attack_rho: float = 0.0) -> float:
    correct = 0
    for x, y in data:
        eps = None
        if attack_rho:
            g = hidden_grad(model, x, y)
            g_norm = norm(g) + 1e-12
            eps = [attack_rho * gi / g_norm for gi in g]
        pred = 1 if model.predict_proba(x, eps=eps) >= 0.5 else 0
        correct += int(pred == y)
    return correct / len(data)


def main() -> None:
    train_data = make_data(seed=0, n=240)
    test_data = make_data(seed=1, n=200)

    normal = TinyModel.init(seed=10)
    vaccine = TinyModel.init(seed=10)

    train_normal(normal, train_data[:], epochs=90, lr=0.45)
    train_vaccine_style(vaccine, train_data[:], epochs=90, lr=0.45, rho=0.35)

    print("Tiny Vaccine demo")
    print("=================")
    print("Clean accuracy:")
    print(f"  normal training : {accuracy(normal, test_data) * 100:5.1f}%")
    print(f"  vaccine-style   : {accuracy(vaccine, test_data) * 100:5.1f}%")
    print("Accuracy under hidden perturbation:")
    print(f"  normal training : {accuracy(normal, test_data, attack_rho=0.55) * 100:5.1f}%")
    print(f"  vaccine-style   : {accuracy(vaccine, test_data, attack_rho=0.55) * 100:5.1f}%")
    print()
    print("What to notice:")
    print("- Vaccine-style training is not magic; it trains on a harder perturbed version of the hidden state.")
    print("- The real project applies this idea inside attention modules of LLMs with PyTorch hooks.")
    print("- In the real code, rho controls the perturbation strength.")


if __name__ == "__main__":
    main()

