"""Core DSD-LI-PO implementation.

This file contains the main update logic of DSD-LI-PO: the baseline PO behavioral
update, dynamic swarm dimension hybridization (DSD), stagnation-triggered lens
imaging (LI), and replacement of the worst individual by the LI candidate.

The release is intentionally concise and does not include full benchmark drivers,
statistical analysis scripts, or result-generation pipelines.
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


def dsd_li_po(
    population_size: int,
    max_iter: int,
    lb: float | Array,
    ub: float | Array,
    dim: int,
    objective: Objective,
    cr_max: float = 0.4,
    cr_min: float = 0.05,
    pd0: float = 0.2,
    gamma: float = 0.5,
    eta: float = 10.0,
    stagnation_limit: int = 15,
) -> Tuple[Array, Array, float]:
    """Run DSD-LI-PO on a minimization objective.

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
    cr_max, cr_min:
        Initial and final individual-level DSD hybridization probabilities.
    pd0:
        Initial dimension-level inheritance probability coefficient.
    gamma, eta:
        Nonlinear LI scaling parameters.
    stagnation_limit:
        Number of consecutive non-improving iterations before LI is activated.

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
    stagnation_counter = 0

    for iteration in range(max_iter):
        progress = iteration / max_iter
        cr = cr_max - (cr_max - cr_min) * progress
        pd = pd0 * (1.0 - progress)

        alpha = np.random.rand() / 5.0
        theta = np.random.rand() * math.pi
        mean_position = np.mean(population, axis=0)
        improved_this_iteration = False

        new_population = population.copy()
        new_fitness = np.empty(population_size, dtype=float)

        for i in range(population_size):
            behavior = np.random.randint(1, 5)

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

            # Dynamic swarm dimension hybridization (DSD).
            if np.random.rand() < cr:
                mask = np.random.rand(dim) < pd
                candidate[mask] = best_solution[mask]

            candidate = np.clip(candidate, lb_arr, ub_arr)
            value = float(objective(candidate))
            new_population[i] = candidate
            new_fitness[i] = value

            if value < best_fitness:
                best_fitness = value
                best_solution = candidate.copy()
                improved_this_iteration = True

        if improved_this_iteration:
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Stagnation-triggered contracted lens imaging (LI).
        if stagnation_counter >= stagnation_limit:
            center = (lb_arr + ub_arr) / 2.0
            k = (1.0 + progress ** gamma) * eta
            li_candidate = center + (center - best_solution) / k
            li_candidate = np.clip(li_candidate, lb_arr, ub_arr)
            li_fitness = float(objective(li_candidate))

            if li_fitness < best_fitness:
                best_fitness = li_fitness
                best_solution = li_candidate.copy()

            worst_index = int(np.argmax(new_fitness))
            new_population[worst_index] = li_candidate
            new_fitness[worst_index] = li_fitness
            stagnation_counter = 0

        order = np.argsort(new_fitness)
        population, fitness = new_population[order], new_fitness[order]
        curve[iteration] = best_fitness

    return curve, best_solution, best_fitness


# Backward-compatible alias for users who prefer the name used in the manuscript.
DSD_LI_PO = dsd_li_po
