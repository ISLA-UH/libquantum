"""
chirp_gamma.py
Gamma is the sweep index, an essential parameter for Gabor chirps

"""
import numpy as np
import matplotlib.pyplot as plt
print(__doc__)


if __name__ == "__main__":
    q = np.logspace(start=-4, stop=4, num=9**3, base=2)
    gamma = np.sqrt(0.5*(np.sqrt(4*q**2 + 1)-1))
    mu = np.sqrt(1 + gamma**2)

    # Plot figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(q, gamma, label=r'$\gamma$')
    ax.plot(q, q, '--', label=r'$q$')
    ax.plot(q, np.sqrt(q), '-.', label=r'$\sqrt{q}$')
    ax.plot(q, mu, label=r'$\mu$')
    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    ax.legend(loc='center right', fontsize='x-large')
    ax.set_xlabel(r'$q \approx \frac{1}{5} \frac{Q_N}{Q_s}$', fontsize='xx-large')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.set_ylim(gamma[0], gamma[-1]+0.1)
    ax.set_xlim(q[0], q[-1])
    ax.grid(True)
    fig.tight_layout()
    plt.show()
