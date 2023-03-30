import pandas as pd
import numpy as np
import math
import openpyxl

# 当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
np.random.seed(0)
# 根据期望获得reward
def get_reward(u):
    distribution = np.random.normal(u, 0.1, 1000)
    res = np.random.choice(distribution,1)
    res = min(1, res)
    res = max(0, res)
    return res

T = 10000
# 设置源节点数目 以及 期望收益
M = 45
ms = np.random.uniform(low=0, high=1, size=M)
optimal = np.max(ms)

# UCB
def ucb(M, T):
    regret = np.zeros(T+1, dtype=float)
    paoi = np.zeros((T+1,M), dtype=int)
    aoi = np.zeros((T+1,M), dtype=int)

    S = np.zeros(M, dtype=float)# 累积reward
    N = np.zeros(M, dtype=int)# 被选择的次数
    c = np.zeros(M, dtype=float)# 置信半径
    ucb = np.zeros(M, dtype=float)# S/N + c
    for t in range(M):
        r = get_reward(ms[t])
        S[t] += r
        N[t] += 1
        regret[t+1] = regret[t]+max(0,optimal-r)

        for i in range(M):
            aoi[t+1][i] = aoi[t][i] + 1
        aoi[t+1][t] = 0
    for t in range(M, T):
        for i in range(M):
            ucb[i] = S[i]/N[i] + math.sqrt(2 * math.log(t) / N[i])
        index= np.argmax(ucb)
        r = get_reward(ms[index])
        S[index] += r
        N[index] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)

        for i in range(M):
            aoi[t+1][i] = aoi[t][i] + 1
        aoi[t+1][index] = 0
    paoi = np.max(aoi, axis=1)
    return regret, paoi
regret, aoi = ucb(M,T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(regret)
df.to_excel(writer,index=False,sheet_name='ucb_regret')
df = pd.DataFrame(aoi)
df.to_excel(writer,index=False,sheet_name='ucb_aoi')
writer.save()



#设置AoI约束阈值
def aucb(M, T):
    a = 100
    regret = np.zeros(T+1, dtype=float)
    paoi = np.zeros((T+1,M), dtype=int)
    aoi = np.zeros((T+1,M), dtype=int)

    S = np.zeros(M, dtype=float)# 累积reward
    N = np.zeros(M, dtype=int)# 被选择的次数
    c = np.zeros(M, dtype=float)# 置信半径
    ucb = np.zeros(M, dtype=float)# S/N + c
    for t in range(M):
        r = get_reward(ms[t])
        S[t] += r
        N[t] += 1
        regret[t+1] = regret[t]+max(0,optimal-r)

        for i in range(M):
            aoi[t+1][i] = aoi[t][i] + 1
        aoi[t+1][t] = 0
    for t in range(M, T):
        index = 0
        if np.max(aoi, axis=1)[t] > a:
            index = np.argmax(aoi, axis=1)[t]
        else:
            for i in range(M):
                ucb[i] = S[i]/N[i] + math.sqrt(2 * math.log(t) / N[i])
            index= np.argmax(ucb)
        r = get_reward(ms[index])
        S[index] += r
        N[index] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)

        for i in range(M):
            aoi[t+1][i] = aoi[t][i] + 1
        aoi[t+1][index] = 0
    paoi = np.max(aoi, axis=1)
    return regret, paoi
regret, aoi = aucb(M,T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(regret)
df.to_excel(writer,index=False,sheet_name='aucb_regret')
df = pd.DataFrame(aoi)
df.to_excel(writer,index=False,sheet_name='aucb_aoi')
writer.save()



#EXP3
def exp3(M,T,gama=0.07):
    a = 100
    regret = np.zeros(T + 1, dtype=float)
    paoi = np.zeros((T + 1, M), dtype=int)
    aoi = np.zeros((T + 1, M), dtype=int)
    w = np.ones(M, dtype=float)
    pr = np.zeros(M, dtype=float)

    S = np.zeros(M, dtype=float)  # 累积reward
    N = np.zeros(M, dtype=int)  # 被选择的次数

    for t in range(T):
        for i in range(M):
            pr[i] = (1 - gama) * (w[i] / sum(w)) + gama / M
        index = np.random.choice(M, 1, p=pr)

        r = get_reward(ms[index])
        S[index] += r
        N[index] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)
        w[index] *= np.exp(gama * r / M)
        for i in range(M):
            aoi[t + 1][i] = aoi[t][i] + 1
        aoi[t + 1][index] = 0
    paoi = np.max(aoi, axis=1)
    return regret, paoi
exp3_regret,exp3_aoi = exp3(M,T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(exp3_regret)
df.to_excel(writer,index=False,sheet_name='exp3_regret')
df = pd.DataFrame(exp3_aoi)
df.to_excel(writer,index=False,sheet_name='exp3_aoi')
writer.save()

#greedy
def greedy(M, T, epsilon=0.1):
    a = 100
    regret = np.zeros(T + 1, dtype=float)
    paoi = np.zeros((T + 1, M), dtype=int)
    aoi = np.zeros((T + 1, M), dtype=int)

    S = np.zeros(M, dtype=float)  # 累积reward
    N = np.zeros(M, dtype=int)  # 被选择的次数
    values = np.zeros(M, dtype=float)

    for t in range(T):
        for i in range(M):
            if N[i] != 0:
                values[i] = S[i]/N[i]
        z = np.random.random()
        if z > epsilon:
            index = np.argmax(values)
        else:
            index = np.random.randint(0, M)

        r = get_reward(ms[index])
        S[index] += r
        N[index] += 1
        regret[t + 1] = regret[t] + max(0, optimal - r)
        for i in range(M):
            aoi[t + 1][i] = aoi[t][i] + 1
        aoi[t + 1][index] = 0
    paoi = np.max(aoi, axis=1)
    return regret, paoi

greedy_regret, greedy_aoi = greedy(M, T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("main.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(greedy_regret)
df.to_excel(writer,index=False,sheet_name='greedy_regret')
df = pd.DataFrame(greedy_aoi)
df.to_excel(writer,index=False,sheet_name='greedy_aoi')
writer.save()