"""
chirp_gamma.py
Gamma is the sweep index, an essential parameter for Gabor chirps

"""
import numpy as np
import matplotlib.pyplot as plt
print(__doc__)


if __name__ == "__main__":
    gamma = np.logspace(start=-4, stop=4, num=9**3, base=2)
    mu = np.sqrt(1 + gamma**2)

    # Plot figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(gamma, mu)
    ax.xaxis.set_tick_params(labelsize='x-large')
    ax.yaxis.set_tick_params(labelsize='x-large')
    ax.set_xlabel(r'$\gamma \approx \frac{1}{5} \frac{Q_N}{Q_s}$', fontsize='xx-large')
    ax.set_ylabel(r'$\mu = \sqrt{1+\gamma^2}$', fontsize='xx-large')
    ax.set_yscale('log', base=2)
    ax.set_xscale('log', base=2)
    ax.set_ylim(1 - 0.05, mu[-1]+0.01)
    ax.set_xlim(gamma[0], gamma[-1])
    ax.grid(True)
    fig.tight_layout()
    plt.show()
