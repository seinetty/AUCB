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
M = 15
ms = np.random.uniform(low=0, high=1, size=M)
optimal = np.max(ms)




#设置AoI约束阈值
def aucb(M, T):
    A = [40,60,80,100,120,140,160,180,200]
    R = []
    P = []
    for a in A:
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
        paoi = np.max(np.max(aoi, axis=1))
        R.append(regret[T])
        P.append(paoi)

    return R, P
regret, aoi = aucb(M,T)
# 往Excel里追加sheet和数据
writer = pd.ExcelWriter("self.xlsx", mode="a", engine="openpyxl")
df = pd.DataFrame(regret)
df.to_excel(writer,index=False,sheet_name='regret')
df = pd.DataFrame(aoi)
df.to_excel(writer,index=False,sheet_name='aoi')
writer.save()