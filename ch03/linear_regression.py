from scipy.stats import multivariate_normal as mult
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    M = 5       # パラメータ数
    N = 10      # 訓練データ数
    lmd = 0.5   # λ,精度パラメータ
    m = np.random.randn(M) * 0.1  # 平均パラメータ
    Lambda = np.random.randn(M,M) * 0.1 # 共分散行列の逆行列

    np.random.seed(1234)
    # 学習データ 三角関数＋ノイズ f(x) = sin x + ε_n
    X = np.random.randn(N)
    XX = mk(X,M).reshape([-1,N]).T
    Y = np.sin(X) + np.random.randn(N)*0.1

    # 三角関数
    truthX = np.arange(min(X),max(X),0.01)
    truthY = np.sin(truthX)
    plt.plot(truthX,truthY)

    plt.plot(X,Y,'o')

    # 学習
    Lambda_hat = learn_Lambda(XX,Lambda,lmd)
    m_hat = learn_m(XX,Y,Lambda,Lambda_hat,lmd,m)

    print("Lambda_hat = ")
    print(Lambda_hat)
    print("m_hat = ")
    print(m_hat)

    plt.show()

def mk(x,M):
    res = np.array([])
    for i in range(M):
        res = np.append(res,x**i)
    return res

def learn_Lambda(XX,Lambda,lmd):
    res = np.array([])
    for i in range(XX.shape[0]):
        res = np.append(res, dot(XX[i],XX[i]))
    return lmd * np.sum(res) + Lambda

def learn_m(XX,Y,Lambda,Lambda_hat,lmd,m):
    ynxn = np.array([])
    for i in range(XX.shape[0]):
        ynxn = np.append(ynxn, np.dot(Y[i],XX[i]))
    sigma = sum(ynxn)
    tmp = np.dot(lmd, (sigma + np.dot(Lambda, m)))
    return np.dot(np.linalg.inv(Lambda_hat), tmp)

def dot(v,w):
    v = v.reshape([1,-1])
    w = w.reshape([1,-1])
    return np.dot(v.T,w)

if __name__ == "__main__":
    main()
