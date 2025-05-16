import numpy as np
import matplotlib.pyplot as plt


n = 10000
def hipergeometrica(n, ngood, nbad, nsample):
    return np.random.hypergeometric(ngood, nbad, nsample, n)
ngood, nbad, nsample = 500, 50, 100
samples = hipergeometrica(n, ngood, nbad, nsample)

plt.hist(samples, bins=50, alpha=0.5, label='Hipergeometrica')
plt.legend()
plt.show()