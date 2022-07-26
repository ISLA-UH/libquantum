"""
chirp_sweep.py
Show the chirp sweep waveforms various values of gamma

"""
import numpy as np
import matplotlib.pyplot as plt
from libquantum.scales import EPSILON
print(__doc__)


if __name__ == "__main__":

    # Gabor atom specifications
    order = 12
    scale_multiplier = 3/4*np.pi * order
    omega = np.pi/16  # Only matters for adequate sampling. Nyquist is at pi

    # Atoms scale
    scale = scale_multiplier/omega
    # Largest amplitude and information is for the atom

    # Real or imaginary signal
    chirp_sig_type = 'complex'  # 'real', 'imag', or 'complex'

    if chirp_sig_type == 'complex':
        atom_peak_amp = 1./(np.pi*scale**2)**0.25
    else:
        atom_peak_amp = np.sqrt(2)/(np.pi*scale**2)**0.25

    atom_peak_power = atom_peak_amp**2
    atom_peak_info = -atom_peak_power*np.log2(atom_peak_power)

    window_support_points = 2*np.pi*scale
    # scale up
    window_support_pow2 = 2**(np.ceil(np.log2(window_support_points)))

    gamma0 = np.logspace(start=2, stop=-2, num=5, base=2)
    # gamma0 = np.array([0, 1/4, 1/2, 1, 2, 4])
    mu0 = np.sqrt(1 + gamma0**2)
    # Working with number of points
    window_duration = window_support_pow2 * mu0
    window_duration_max = int(np.max(window_duration))
    # Shared time vector
    time0 = np.arange(window_duration_max)
    dt = np.mean(np.diff(time0))
    time = time0 - time0[-1]/2

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex='col', figsize=(8, 6))
    for i, gamma in enumerate(gamma0):
        mu = mu0[i]
        chirp_scale = scale * mu
        chirp_phase = omega * time + 0.5 * gamma * (time / chirp_scale) ** 2
        chirp_wf = np.exp(-0.5 * (time / chirp_scale) ** 2 + 1j * chirp_phase)
        # chirp_phase = omega*time + 0.5*gamma*(time/scale)**2
        # chirp_wf = np.exp(-0.5*(time/scale)**2 + 1j*chirp_phase)

        # Select real or imaginary component
        # Apply theoretical value from -inf to inf
        # 1/2 of total power for real or imaginary part only

        if chirp_sig_type == 'real':
            chirp_wf_sig = np.real(chirp_wf).astype(np.float32)
            chirp_var = 0.5 * (np.pi * chirp_scale ** 2) ** 0.5  # * (1 + np.exp(-chirp_scale*omega))
        elif chirp_sig_type == 'imag':
            chirp_wf_sig = np.real(chirp_wf).astype(np.float32)
            chirp_var = 0.5 * (np.pi * chirp_scale ** 2) ** 0.5  # * (1 - np.exp(-chirp_scale*omega))
        else:
            chirp_wf_sig = chirp_wf
            chirp_var = (np.pi * chirp_scale ** 2) ** 0.5

        # Theoretical rms
        chirp_rms = np.sqrt(chirp_var)

        # Numerical value from std and var
        # chirp_var_sig = np.var(chirp_wf_sig) * len(time)
        chirp_var_sig = np.sum(chirp_wf_sig * chirp_wf_sig.conjugate())
        chirp_rms_sig = np.sqrt(chirp_var_sig)

        # Percent error in var, rms, and information
        percent_error_var = np.abs(chirp_var_sig-chirp_var)/chirp_var*100
        percent_error_rms = np.abs(chirp_rms_sig-chirp_rms)/chirp_rms*100

        chirp_wf_in = chirp_wf_sig / chirp_rms
        chirp_wf_power = np.real(chirp_wf_in * chirp_wf_in.conjugate())

        print('Gamma:', gamma)
        print('Percent difference between theoretical and sig var:', percent_error_var)
        print('Percent difference between theoretical and sig rms:', percent_error_rms)

        # Theoretical value for real and complex components
        chirp_info_theory = 0.5*np.log2(np.pi*np.exp(1)*chirp_scale**2)
        # Instantaneous Shannon information for discrete data
        chirp_wf_info = -chirp_wf_power*np.log2(chirp_wf_power + EPSILON)
        chirp_sum_info = np.sum(chirp_wf_info)
        # chirp_sum_info_theory = 0.5*np.log2(np.pi*np.exp(1)*chirp_scale**2)
        chirp_info_error_bits = chirp_info_theory - chirp_sum_info
        print('Information difference in bits, total (theoretical - sig) info:', chirp_info_error_bits)
        print("For real and imaginary components, this is approximately sqrt(pi)/4")
        correction_factor = np.sqrt(np.pi)/4  # * (1 + np.exp(-1/32*(omega*mu)**2))
        # TODO: increases to ~ np.sqrt(np.pi)/2 as mu -> 1
        print('Correction factor:', correction_factor)
        print('Info dif in bits divided by sqrt(pi)/4:', chirp_info_error_bits/correction_factor)

        # Handle label powers
        gamma_str = str(int(np.log2(gamma0[i])))
        # gamma_str = r'$2^{}$'

        info_str = str(np.around(chirp_sum_info, decimals=1))
        ax1.plot(time/window_support_points, chirp_wf_info, label=r'$\sum = $' + info_str)
        ax1.xaxis.set_tick_params(labelsize='x-large')
        ax1.yaxis.set_tick_params(labelsize='x-large')
        ax1.set_ylabel('Info, bits', rotation=90, fontsize='xx-large')
        ax1.set_yscale('log', base=2)
        # ax.set_xscale('log', base=2)
        ax1.set_ylim(2**-16, 2**np.ceil(np.log2(1.1*atom_peak_info)))
        ax1.set_xlim(-2, 2)

        if chirp_sig_type == 'complex':
            ax2.plot(time/window_support_points, np.real(chirp_wf_in), label=r'$ln_2\gamma = $' + gamma_str)
            # ax2.plot(time/window_support_points, np.imag(chirp_wf_in), '--')
        else:
            ax2.plot(time/window_support_points, chirp_wf_in, label=r'$\gamma = $' + gamma_str)
        ax2.xaxis.set_tick_params(labelsize='x-large')
        ax2.yaxis.set_tick_params(labelsize='x-large')
        ax2.set_xlabel(r'$\dfrac{time}{M_N}$', fontsize='xx-large')
        ax2.set_ylabel(r'$\dfrac{f}{f_{rms}}$', rotation=0, fontsize='xx-large')
        ax2.set_ylim(-1.1*atom_peak_amp, 1.1*atom_peak_amp)
        ax2.set_xlim(-2, 2)
    ax1.legend(loc='upper right', fontsize='large')
    ax1.grid(True)
    ax2.legend(loc='upper right', fontsize='x-large')
    ax2.grid(True)
    fig.tight_layout()

    plt.show()
