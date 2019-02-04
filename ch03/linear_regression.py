from scipy.stats import multivariate_normal as mult
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    np.random.seed(1234)

    M = 5      # パラメータ数
    N = 30      # 訓練データ数
    lmd = 3   # λ,精度パラメータ
    m = np.random.randn(M).T * 0.01  # 平均パラメータ
    Lambda = np.random.randn(M,M).T * 0.01 # 共分散行列の逆行列

    # 学習データ 三角関数＋ノイズ f(x) = sin x + ε_n
    X = np.random.rand(N)*6.28
    XX = mk(X,M).reshape([-1,N]).T
    Y = np.sin(X) + np.random.randn(N)*0.1

    # 三角関数
    truthX = np.arange(min(X)-0.5,max(X)+0.5,0.01)
    truthY = np.sin(truthX)
    plt.plot(truthX,truthY,label="sin(x)")

    plt.plot(X,Y,'o',label="train data")

    # 学習
    Lambda_hat = learn_Lambda(XX,Lambda,lmd)
    m_hat = learn_m(XX,Y,Lambda,Lambda_hat,lmd,m)

    mu_aster = np.array([])
    inv_lmd_aster = np.array([])
    for i in range(truthX.shape[0]):
        mu_aster = np.append(mu_aster,calc_mu_aster(m_hat,truthX[i]))
#        import pdb; pdb.set_trace()
        inv_lmd_aster = np.append(inv_lmd_aster,calc_inv_lmd_aster(Lambda_hat,lmd,truthX[i],M))

    print(mu_aster.shape)
    print(inv_lmd_aster.shape)

    plt.plot(truthX,mu_aster,label="pred mean")
    plt.plot(truthX,mu_aster+np.sqrt(inv_lmd_aster),label="pred devi+")
    plt.plot(truthX,mu_aster-np.sqrt(inv_lmd_aster),label="pred devi-")
    plt.legend(loc='upper left')
    plt.show()

def mk(x,M):
    res = np.array([])
    for i in range(M):
        res = np.append(res,x**i)
    return res

def learn_Lambda(XX,Lambda,lmd):
    xnxn = np.zeros((XX.shape[1],XX.shape[1],))
    for i in range(XX.shape[0]):
        xnxn = xnxn + dot(XX[i],XX[i])
    return lmd * xnxn + Lambda

def learn_m(XX,Y,Lambda,Lambda_hat,lmd,m):
    ynxn = np.zeros(XX.shape[1])
    for i in range(XX.shape[0]):
        ynxn = ynxn + Y[i]*XX[i]
    tmp = lmd * ynxn + np.dot(Lambda, m)
    return np.dot(np.linalg.inv(Lambda_hat), tmp)

def calc_mu_aster(m_hat,x_aster):
    xx_aster = mk(x_aster,m_hat.shape[0])
    return np.dot(m_hat.T, xx_aster)

def calc_inv_lmd_aster(Lambda_hat,lmd,x_aster,M):
    xx_aster = mk(x_aster,M)
    return lmd**(-1) + np.dot(np.dot(xx_aster, np.linalg.inv(Lambda_hat)),xx_aster.T)

def dot(v,w):
    v = v.reshape([1,-1])
    w = w.reshape([1,-1])
    return np.dot(v.T,w)

if __name__ == "__main__":
    main()
