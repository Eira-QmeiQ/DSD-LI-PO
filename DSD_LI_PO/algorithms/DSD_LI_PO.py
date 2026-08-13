"""
Dynamic Swarm Dimension Hybridization and Lens Imaging Parrot Optimizer (DSD-LI-PO)

Core implementation of the proposed optimizer.
"""

import numpy as np
import math


def DSD_LI_PO(N, Max_iter, lb, ub, dim, fobj):
    if np.isscalar(ub): ub, lb = np.ones(dim) * ub, np.ones(dim) * lb
    X = np.random.rand(N, dim) * (ub - lb) + lb
    fitness = np.array([fobj(x) for x in X])
    idx = np.argsort(fitness)
    fitness, X = fitness[idx], X[idx]
    GBestF, GBestX = fitness[0], X[0, :].copy()
    curve = np.zeros(Max_iter)
    stagnation_counter, stagnation_limit = 0, 15  # 缩短停滞阈值

    for t in range(Max_iter):
        # 核心逻辑：动态调整策略权重
        # 杂交率随迭代从 0.4 线性降至 0.05，保护后期多样性
        CR = 0.4 - (0.35 * (t / Max_iter))

        alpha, sita, updated_flag = np.random.rand() / 5, np.random.rand() * np.pi, False
        X_new, fitness_new = X.copy(), np.zeros(N)

        for j in range(N):
            St = np.random.randint(1, 5)
            # PO 基础行为
            if St == 1:
                X_new[j, :] = (X[j, :] - GBestX) * levy(dim) + np.random.rand() * np.mean(X, 0) * (
                            1 - t / Max_iter) ** (2 * t / Max_iter)
            elif St == 2:
                X_new[j, :] = X[j, :] + GBestX * levy(dim) + np.random.rand() * np.ones(dim)
            elif St == 3:
                if np.random.rand() < 0.5:
                    X_new[j, :] = X[j, :] + alpha * (1 - t / Max_iter) * (X[j, :] - np.mean(X, 0))
                else:
                    X_new[j, :] = X[j, :] + alpha * (1 - t / Max_iter) * np.exp(
                        -j / (np.random.rand() * Max_iter + 1e-10))
            else:
                X_new[j, :] = X[j, :] + np.random.rand() * np.cos((np.pi * t) / (2 * Max_iter)) * (
                            GBestX - X[j, :]) - np.cos(sita) * (t / Max_iter) ** (2 / Max_iter) * (X[j, :] - GBestX)

            # --- 优化策略 A: 动态维度杂交 ---
            if np.random.rand() < CR:
                mask = np.random.rand(dim) < (0.2 * (1 - t / Max_iter))  # 杂交维度数也随之衰减
                X_new[j, mask] = GBestX[mask]

            X_new[j, :] = np.clip(X_new[j, :], lb, ub)
            fitness_new[j] = fobj(X_new[j, :])

            if fitness_new[j] < GBestF:
                GBestF, GBestX, updated_flag = fitness_new[j], X_new[j, :].copy(), True

        # --- 优化策略 B: 增强型动态透镜成像 ---
        if not updated_flag:
            stagnation_counter += 1
        else:
            stagnation_counter = 0

        if stagnation_counter >= stagnation_limit:
            # 引入非线性指数 $k$，增加后期的跳动力度
            k = (1 + (t / Max_iter) ** 0.5) * 10
            X_LI = np.clip((lb + ub) / 2 + (lb + ub) / (2 * k) - GBestX / k, lb, ub)
            f_LI = fobj(X_LI)
            if f_LI < GBestF:
                GBestF, GBestX = f_LI, X_LI.copy()
            # Inject the LI candidate by replacing the worst individual
            worst_idx = np.argmax(fitness_new)
            X_new[worst_idx, :] = X_LI
            fitness_new[worst_idx] = f_LI
            stagnation_counter = 0

        X, fitness = X_new.copy(), fitness_new.copy()
        idx = np.argsort(fitness)
        fitness, X = fitness[idx], X[idx]
        curve[t] = GBestF

    return curve, GBestX, GBestF


def levy(d):
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return np.random.randn(d) * sigma / np.power(np.abs(np.random.randn(d)), 1 / beta)