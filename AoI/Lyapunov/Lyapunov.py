import pandas as pd
import numpy as np
import math
import openpyxl

# 当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
np.random.seed(0)


# 根据期望获得reward
def get_reward(u):
    distribution = np.random.normal(u, 0.1, 1000)
    res = np.random.choice(distribution, 1)
    res = min(1, res)
    res = max(0, res)
    return res


T = 10000
# 设置源节点数目 以及 期望收益
M = 45
ms = np.random.uniform(low=0, high=1, size=M)
optimal = np.max(ms)

# 1 50 100 500 1000

# 设置AoI约束阈值
def aucb(M, T):
    a = 100
    regret = np.zeros(T + 1, dtype=float)
    paoi = np.zeros((T + 1, M), dtype=int)
    aoi = np.zeros((T + 1, M), dtype=int)

    S = np.zeros(M, dtype=float)  # 累积reward
    N = np.zeros(M, dtype=int)  # 被选择的次数
    c = np.zeros(M, dtype=float)  # 置信半径
    ucb = np.zeros(M, dtype=float)  # S/N + c
    for t in range(M):
        r = get_reward(ms[t])
        S[t] += r
        N[t] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)

        for i in range(M):
            aoi[t + 1][i] = aoi[t][i] + 1
        aoi[t + 1][t] = 0
    for t in range(M, T):
        index = 0
        if np.max(aoi, axis=1)[t] > a:
            index = np.argmax(aoi, axis=1)[t]
        else:
            for i in range(M):
                ucb[i] = S[i] / N[i] + math.sqrt(2 * math.log(t) / N[i])
            index = np.argmax(ucb)
        r = get_reward(ms[index])
        S[index] += r
        N[index] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)

        for i in range(M):
            aoi[t + 1][i] = aoi[t][i] + 1
        aoi[t + 1][index] = 0
    paoi = np.max(aoi, axis=1)
    return regret, paoi


regret, aoi = aucb(M, T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(regret)
df.to_excel(writer, index=False, sheet_name='aucb_regret')
df = pd.DataFrame(aoi)
df.to_excel(writer, index=False, sheet_name='aucb_aoi')
writer.save()


def lyapunov(V, M, T):
    a = 100
    A_MAX = a
    regret = np.zeros(T + 1, dtype=float)
    paoi = np.zeros((T + 1, M), dtype=int)
    aoi = np.zeros((T + 1, M), dtype=int)

    X = np.zeros((T + 1, M), dtype=int)

    """遍历选择不同节点的"""
    for t in range(0, T):
        index = 0
        L = 100000000
        for i in range(M):
            r = get_reward(ms[i])
            l1 = - V*r - X[t][i] * (aoi[t][i] + 1 - A_MAX)
            for j in range(M):
                l1 = l1 + X[t][j] * (aoi[t][j] + 1 - A_MAX)
                if l1 < L:
                    L = l1
                    index = i

        r = get_reward(ms[index])
        regret[t + 1] = regret[t] + max(0, optimal - r)
        for i in range(M):
            aoi[t + 1][i] = aoi[t][i] + 1
        aoi[t + 1][index] = 0
        for i in range(M):
            X[t + 1][i] = max(X[t][i] - A_MAX, 0) + aoi[t][i]
    paoi = np.max(aoi, axis=1)
    return regret, paoi


Vs = [1, 1000]


for v in Vs:
    regret, aoi = lyapunov(v, M, T)
    # 往Excel里追加sheet和数据
    writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
    df = pd.DataFrame(regret)
    df.to_excel(writer, index=False, sheet_name='lyapunov_regret'+str(v))
    df = pd.DataFrame(aoi)
    df.to_excel(writer, index=False, sheet_name='lyapunov_aoi'+str(v))
    writer.save()


