from scipy.stats import multivariate_normal as mult
from scipy.stats import norm
from scipy.stats import rv_discrete
import numpy as np
import matplotlib.pyplot as plt

K = 3

sk = np.arange(K)
pk = (0.3,0.6,0.1)

cat = rv_discrete(name='cat', values=(sk,pk))
x = cat.rvs(size=100)
