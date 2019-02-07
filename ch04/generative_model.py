from scipy.stats import multivariate_normal as mult
from scipy.stats import norm
from scipy.stats import rv_discrete
import numpy as np
import matplotlib.pyplot as plt

K = 3
N = 1000

mean = [ np.array([0,0])
       , np.array([10,3])
       , np.array([5,9])]

cov = [ np.array([[1,0],[0,2]])
      , np.array([[5,1.5],[1,2]])
      , np.array([[1,2],[2,7]])]

vk = np.arange(K)
pk = (0.3,0.4,0.3)
sk = np.zeros((N,K))

cat = rv_discrete(name='cat', values=(vk,pk))
xs = []
ys = []
for i in range(N):
    sk = cat.rvs()
    (x, y) = mult.rvs(mean=mean[sk],cov=cov[sk])
    xs = xs + [x]
    ys = ys + [y]

plt.scatter(xs,ys)
plt.show()
