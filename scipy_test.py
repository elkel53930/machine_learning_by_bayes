from scipy.stats import binom
from scipy import stats
import math
import matplotlib.pyplot as plt

import numpy as np

#乱数のseedを設定
np.random.seed(1234)

M = 10
m = np.arange(0,M,1)
mu = 0.3

bi = binom.pmf(m,M,mu)

plt.bar(m,bi)
plt.show()
