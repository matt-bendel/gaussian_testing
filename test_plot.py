import matplotlib.pyplot as plt
import numpy as np

covs_rcgan = [0.0077 / 10, 0.0251 / 20, 0.0817 / 30, 0.1892 / 40, 0.3786 / 50, 0.6000 / 60, 0.8251 / 70, 1.1682 / 80, 1.5210 / 90, 1.7432 / 100]
covs_rcgan_lazy = [0.0046 / 10, 0.0096 / 20, 0.0191 / 30, 0.0319 / 40, 0.0596 / 50, 0.0817 / 60, 0.1347 / 70, 0.1460 / 80, 0.2389 / 90, 0.2662 / 100]
covs_theory = [1.74e-6 / 10, 5.14e-6 / 20, 9.53e-6 / 30, 2.22e-5 / 40, 3.18e-5 / 50, 4.60e-5 / 60, 6.03e-5 / 70, 8.98e-5 / 80, 1.04e-4 / 90, 1.26e-4 / 100]
covs_rcgan_lazy_freqs = [0.2100, 0.2139, 0.2662, 0.2766, 0.3310]
covs_rcgan_lazy_k = [0.3500, 0.3269, 0.3211, 0.2521, 0.2139]
d_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
freq_vals = [1, 50, 100, 150, 200]

k_vals = [5, 10, 25, 50, 100]

plt.plot(d_vals, covs_rcgan)
plt.plot(d_vals, covs_rcgan_lazy)
plt.plot(d_vals, covs_theory)
plt.xlabel('d')
plt.ylabel('CFID')
plt.title('CFID / d vs. d')
plt.legend(['rcGAN', 'EigenGAN', 'Theory'])
plt.savefig('cfid_v_d.png')

plt.figure()
plt.scatter(freq_vals, covs_rcgan_lazy_freqs)
plt.xlabel('Period')
plt.ylabel('CFID')
plt.title('CFID vs. Period for 100d Gaussian data')
plt.ylim([0, 0.5])
plt.xlim([0, 250])
plt.savefig('cfid_v_freq.png')

# TODO: Log spacing in K
# TODO: Train k=5 model
plt.figure()
plt.scatter(k_vals, covs_rcgan_lazy_k)
plt.xlabel('K')
plt.ylabel('CFID')
plt.title('CFID vs. K for 100d Gaussian data')
plt.ylim([0, 0.5])
plt.xlim([0, 150])
plt.savefig('cfid_v_k.png')