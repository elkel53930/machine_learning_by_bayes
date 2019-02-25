from scipy.stats import rv_discrete
from scipy.stats import gamma
from scipy.stats import poisson as poi
from scipy.stats import dirichlet as dir
import matplotlib.pyplot as plt
import numpy as np

K = 3
N = 100
ITERMAX = 200
X = []
a = 1
b = 1
Alpha = np.random.rand(K)
truth_s = []

def main():
    X = generate_data()

    # initialize lambda and pi
    lambda_sample = np.random.rand(K)
    pi_sample = np.random.rand(K)
    pi_sample = pi_sample / np.sum(pi_sample)

    s_samples = np.zeros(N) # shape == (ITERMAX, N)
    lambda_samples = np.zeros(K) # shape == (ITERMAX, K)
    pi_samples = np.zeros(K) # shape == (ITERMAX, K)

    for iter in range(ITERMAX):
        s_sample = np.array([])
        for i in range(N):
            s_sample = np.append(s_sample, sampling_s_n(X[i],lambda_sample,pi_sample))

        lambda_sample = np.array([])
        for k in range(K):
            lambda_sample = np.append(lambda_sample, sampling_lambda_k(X,s_sample,k))

        pi_sample = sampling_pi(s_sample)

        s_samples = np.vstack((s_samples, s_sample))
        pi_samples = np.vstack((pi_samples, pi_sample))
        lambda_samples = np.vstack((lambda_samples, lambda_sample))

        if iter % 50 == 0:
            print("iter = " + str(iter))
            print("estimated lambda = " + str(calc_estimated_lambda(lambda_samples,int(iter/4))))

    estimated_lambda = calc_estimated_lambda(lambda_samples,int(ITERMAX/4))
    s_samples = np.delete(s_samples,slice(int(ITERMAX/4)),0)
    table = np.argsort(estimated_lambda)
    for i in range(len(truth_s)):
        truth_s[i] = table[truth_s[i]]

    s_mean = np.mean(s_samples,axis = 0)
    print(truth_s)
    print(s_mean)
    print("acc = " + str(np.sum(np.abs(s_mean-truth_s))))

#    plt.hist(X,range=(0,60),bins = 61)
#    plt.show()

def calc_estimated_lambda(lambdas, ignore):
    return np.mean(np.delete(lambdas,slice(ignore),0),axis=0)

def generate_data():
    v_k = np.arange(K)
    p_k = (0.2,0.4,0.4)
    lambda_k = (2,18,50)
    cat = rv_discrete(name='cat', values=(v_k,p_k))

    X = np.array([])
    for i in range(N):
        sk = cat.rvs()
        X = np.append(X,poi.rvs(mu=lambda_k[sk]))
        truth_s.append(sk)
    return X

def calc_eta_n_k(x_n, lambda_k, pi_k):
    return np.exp( x_n * np.log(lambda_k) - lambda_k + np.log(pi_k))

def sampling_s_n(x_n, Lambda, Pi):
    eta_n = np.array([])
    for k in range(K):
        eta_n = np.append(eta_n,calc_eta_n_k(x_n, Lambda[k], Pi[k]))
    eta_n = eta_n / np.sum(eta_n)
    v_k = np.arange(K)
    cat = rv_discrete(name='cat', values=(v_k,eta_n))
    return cat.rvs()

def calc_a_k_hat(X, s, k):
    return np.sum(X[s==k]) + a

def calc_b_k_hat(s, k):
    return np.sum(s==k) + a

def sampling_lambda_k(X, s, k):
    a_k_hat = calc_a_k_hat(X,s,k)
    b_k_hat = calc_b_k_hat(s,k)
    return gamma.rvs(a_k_hat, scale = 1/b_k_hat)

def calc_alpha_k_hat(s, k):
    return np.sum(s==k) + Alpha[k]

def sampling_pi(s):
    alpha_k_hat = np.array([])
    for k in range(K):
        alpha_k_hat = np.append(alpha_k_hat,calc_alpha_k_hat(s, k))
    return dir.rvs(alpha_k_hat)[0]

if __name__ == "__main__":
    main()
