"""
chirp_sweep.py
Show the chirp sweep waveforms various values of gamma

"""
import numpy as np
import matplotlib.pyplot as plt
print(__doc__)


if __name__ == "__main__":

    # Gabor atom specifications
    order = 6
    scale_multiplier = 3/4*np.pi * order
    omega = np.pi/16  # Only matters for adequate sampling. Nyquist is at pi

    scale = scale_multiplier/omega
    window_support_points = 2*np.pi*scale
    # scale up
    window_support_pow2 = 2**(np.ceil(np.log2(window_support_points)))

    q = np.logspace(start=4, stop=-2, num=7, base=2)
    gamma0 = np.sqrt(0.5*(np.sqrt(4*q**2 + 1)-1))
    mu0 = np.sqrt(1 + gamma0**2)
    # Working with number of points
    window_duration = window_support_pow2 * mu0
    window_duration_max = int(np.max(window_duration))
    # Shared time vector
    time0 = np.arange(window_duration_max)
    time = time0 - time0[-1]/2

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, gamma in enumerate(gamma0):
        mu = mu0[i]
        cscale = scale*mu
        chirp_phase = omega*time + 0.5*gamma*(time/cscale)**2
        chirp_wf = np.exp(-0.5*(time/cscale)**2 + 1j*chirp_phase)
        # chirp_phase = omega*time + 0.5*gamma*(time/scale)**2
        # chirp_wf = np.exp(-0.5*(time/scale)**2 + 1j*chirp_phase)

        ax.plot(time/scale_multiplier, np.real(chirp_wf), label=str(int(np.log2(q[i]))))
    plt.legend(loc='upper right')
    plt.show()

    # # Plot figure
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(q, gamma, label=r'$\gamma$')
    # ax.plot(q, q, '--', label=r'$q$')
    # ax.plot(q, np.sqrt(q), '-.', label=r'$\sqrt{q}$')
    # ax.plot(q, mu, label=r'$\mu$')
    # ax.xaxis.set_tick_params(labelsize='x-large')
    # ax.yaxis.set_tick_params(labelsize='x-large')
    # ax.legend(loc='center right', fontsize='x-large')
    # ax.set_xlabel(r'$q \approx \frac{1}{5} \frac{Q_N}{Q_s}$', fontsize='xx-large')
    # ax.set_yscale('log', base=2)
    # ax.set_xscale('log', base=2)
    # ax.set_ylim(gamma[0], gamma[-1]+0.1)
    # ax.set_xlim(q[0], q[-1])
    # ax.grid(True)
    # fig.tight_layout()
    # plt.show()
