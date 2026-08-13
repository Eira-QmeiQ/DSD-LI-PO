"""
Parrot Optimizer (PO)

Core implementation used as the baseline optimizer in the DSD-LI-PO study.
"""

import numpy as np
import math


def PO(N, Max_iter, lb, ub, dim, fobj):
    """
    Parrot Optimizer (PO) - 完全对标原版 MATLAB version 2.0 源码

    """
    # 边界处理
    if np.isscalar(ub):
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    # 初始化种群
    X = np.random.rand(N, dim) * (ub - lb) + lb
    fitness = np.zeros(N)

    # 计算初始适应度并排序
    for i in range(N):
        fitness[i] = fobj(X[i, :])

    index = np.argsort(fitness)
    fitness = np.sort(fitness)
    X = X[index, :]

    GBestF = fitness[0]
    GBestX = X[0, :].copy()

    curve = np.zeros(Max_iter)

    # 主循环
    for i in range(Max_iter):
        alpha = np.random.rand() / 5
        sita = np.random.rand() * np.pi

        # 预分配新种群容器 (原版 MATLAB 中使用的是全局更新)
        X_new = X.copy()
        fitness_new = np.zeros(N)

        for j in range(N):
            # 随机选择行为策略 (1-4)
            St = np.random.randint(1, 5)

            # Behavior 1: Foraging
            if St == 1:
                X_new[j, :] = (X[j, :] - GBestX) * levy(dim) + \
                              np.random.rand() * np.mean(X, axis=0) * (1 - i / Max_iter) ** (2 * i / Max_iter)

            # Behavior 2: Staying
            elif St == 2:
                X_new[j, :] = X[j, :] + GBestX * levy(dim) + np.random.rand() * np.ones(dim)

            # Behavior 3: Communicating
            elif St == 3:
                H = np.random.rand()
                if H < 0.5:
                    X_new[j, :] = X[j, :] + alpha * (1 - i / Max_iter) * (X[j, :] - np.mean(X, axis=0))
                else:
                    X_new[j, :] = X[j, :] + alpha * (1 - i / Max_iter) * np.exp(
                        -j / (np.random.rand() * Max_iter + 1e-10))

            # Behavior 4: Fear of strangers
            else:
                X_new[j, :] = X[j, :] + np.random.rand() * np.cos((np.pi * i) / (2 * Max_iter)) * (GBestX - X[j, :]) - \
                              np.cos(sita) * (i / Max_iter) ** (2 / Max_iter) * (X[j, :] - GBestX)

            # 边界控制 (原版 MATLAB 的逻辑点更新)
            X_new[j, :] = np.clip(X_new[j, :], lb, ub)

            # 计算新适应度并实时更新全局最优
            fitness_new[j] = fobj(X_new[j, :])
            if fitness_new[j] < GBestF:
                GBestF = fitness_new[j]
                GBestX = X_new[j, :].copy()

        # 代更替与重排序
        X = X_new.copy()
        fitness = fitness_new.copy()
        index = np.argsort(fitness)
        fitness = np.sort(fitness)
        X = X[index, :]

        curve[i] = GBestF

    return curve, GBestX, GBestF


def levy(d):
    """Levy 飞行函数 - 对标 MATLAB 版实现"""
    beta = 1.5
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn(d) * sigma
    v = np.random.randn(d)
    return u / np.power(np.abs(v), 1 / beta)