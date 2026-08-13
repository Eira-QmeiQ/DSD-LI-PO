# DSD-LI-PO

This repository provides the core implementation of the DSD-LI-PO optimizer and the baseline Parrot Optimizer (PO).

## Algorithms

- `algorithms/PO.py`: baseline Parrot Optimizer
- `algorithms/DSD_LI_PO.py`: proposed DSD-LI-PO optimizer

## Main components

DSD-LI-PO integrates:

1. Dynamic swarm dimension hybridization (DSD)
2. Stagnation-triggered variable precision lens imaging (LI)

## Function interface

Both optimizers use the interface:

```python
curve, best_solution, best_fitness = optimizer(N, Max_iter, lb, ub, dim, fobj)
```

## Requirements

- Python >= 3.9
- NumPy

More experiment scripts and benchmark functions can be added for full reproduction of the manuscript experiments.
