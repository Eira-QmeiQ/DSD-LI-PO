"""Core Parrot Optimizer (PO) implementation.

This file provides a compact baseline PO implementation for method illustration.
It is intended for academic viewing of the main update logic rather than a full
experimental reproduction pipeline.
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np

Array = np.ndarray
Objective = Callable[[Array], float]


def _as_bounds(lb: float | Array, ub: float | Array, dim: int) -> Tuple[Array, Array]:
    """Convert scalar or vector bounds to arrays."""
    lb_arr = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    if lb_arr.shape != (dim,) or ub_arr.shape != (dim,):
        raise ValueError("lb and ub must be scalars or arrays of shape (dim,).")
    return lb_arr, ub_arr


def levy_flight(dim: int, beta: float = 1.5) -> Array:
    """Generate a Levy-flight step using Mantegna's algorithm."""
    sigma = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return u / np.power(np.abs(v) + 1e-12, 1.0 / beta)


def parrot_optimizer(
    population_size: int,
    max_iter: int,
    lb: float | Array,
    ub: float | Array,
    dim: int,
    objective: Objective,
) -> Tuple[Array, Array, float]:
    """Run the baseline Parrot Optimizer.

    Parameters
    ----------
    population_size:
        Number of candidate solutions.
    max_iter:
        Maximum number of iterations.
    lb, ub:
        Lower and upper bounds. Each can be a scalar or a vector of length ``dim``.
    dim:
        Search-space dimension.
    objective:
        Minimization objective function.

    Returns
    -------
    curve:
        Best fitness value recorded at each iteration.
    best_solution:
        Best solution found.
    best_fitness:
        Best objective value found.
    """
    lb_arr, ub_arr = _as_bounds(lb, ub, dim)

    population = np.random.rand(population_size, dim) * (ub_arr - lb_arr) + lb_arr
    fitness = np.array([objective(x) for x in population], dtype=float)
    order = np.argsort(fitness)
    population, fitness = population[order], fitness[order]

    best_fitness = float(fitness[0])
    best_solution = population[0].copy()
    curve = np.empty(max_iter, dtype=float)

    for iteration in range(max_iter):
        alpha = np.random.rand() / 5.0
        theta = np.random.rand() * math.pi
        mean_position = np.mean(population, axis=0)

        new_population = population.copy()
        new_fitness = np.empty(population_size, dtype=float)

        for i in range(population_size):
            behavior = np.random.randint(1, 5)
            progress = iteration / max_iter

            if behavior == 1:  # foraging
                candidate = (
                    (population[i] - best_solution) * levy_flight(dim)
                    + np.random.rand() * mean_position * (1.0 - progress) ** (2.0 * progress)
                )
            elif behavior == 2:  # staying
                candidate = population[i] + best_solution * levy_flight(dim) + np.random.rand(dim)
            elif behavior == 3:  # communication
                if np.random.rand() < 0.5:
                    candidate = population[i] + alpha * (1.0 - progress) * (population[i] - mean_position)
                else:
                    candidate = population[i] + alpha * (1.0 - progress) * np.exp(
                        -i / (np.random.rand() * max_iter + 1e-12)
                    )
            else:  # fear response
                candidate = (
                    population[i]
                    + np.random.rand() * np.cos((math.pi * iteration) / (2.0 * max_iter))
                    * (best_solution - population[i])
                    - np.cos(theta) * progress ** (2.0 / max_iter)
                    * (population[i] - best_solution)
                )

            candidate = np.clip(candidate, lb_arr, ub_arr)
            value = float(objective(candidate))
            new_population[i] = candidate
            new_fitness[i] = value

            if value < best_fitness:
                best_fitness = value
                best_solution = candidate.copy()

        order = np.argsort(new_fitness)
        population, fitness = new_population[order], new_fitness[order]
        curve[iteration] = best_fitness

    return curve, best_solution, best_fitness


# Backward-compatible alias for users who prefer the name used in the manuscript.
PO = parrot_optimizer
