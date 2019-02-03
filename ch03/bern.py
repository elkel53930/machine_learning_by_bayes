from scipy.stats import beta
from scipy.stats import bernoulli as bern
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 学習データ生成
    batch_number = 4
    train_size = 400
    batch_size =  int(train_size / batch_number)
    mu = 0.7 # 実際のμの値
    X = bern.rvs(mu,size = train_size)
    X = X.reshape([batch_number,-1])

    # 事前分布はBeta分布(ベルヌーイ分布の共役事前分布)で与える。
    a, b = 0.5, 0.5
    plot_beta(a,b,"Prior")
    # 学習
    for i in range(batch_number):
        a = learn_a(a,X[i])
        b = learn_b(b,X[i])
        plot_beta(a,b,label='N=' + str((i+1)*batch_size))

    plt.legend(loc='upper left')
    plt.show()

def plot_beta(a,b,label=""):
    xaxis = np.arange(0,1,0.001)
    post = beta.pdf(xaxis,a,b)
    plt.plot(xaxis,post,label=label)

def learn_a(a,X):
    return sum(X) + a

def learn_b(b,X):
    return X.shape[0] - sum(X) + b

if __name__ == "__main__":
    main()
