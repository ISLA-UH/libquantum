import pyfftw
import numpy as np
from scipy.fft import fft, rfft
import matplotlib.pyplot as plt


def chirp_sig(order: float, omega: float, gamma: float, gauss: bool=True):
    """See benchmark signals quantum_chirp"""

    if omega >= 0.8*np.pi:
        print("Omega >= 0.8*pi (AA*Nyquist), reset to pi * 2**(-2/N)")
        omega = np.pi * 2**(-1/order)

    # Gabor atom specifications
    scale_multiplier = 3/4 * np.pi * order

    # Atom scale
    scale = scale_multiplier/omega

    # Chirp index gamma, blueshift
    mu = np.sqrt(1 + gamma**2)
    chirp_scale = scale * mu

    # scale multiplier Mc
    window_support_points = 2*np.pi*chirp_scale
    # scale up
    window_support_pow2 = 2**int((np.ceil(np.log2(window_support_points))))

    time0 = np.arange(window_support_pow2)
    time = time0 - time0[-1]/2

    chirp_phase = omega * time + 0.5 * gamma * (time / chirp_scale) ** 2
    if gauss:
        chirp_wf = np.exp(-0.5 * (time / chirp_scale) ** 2 + 1j * chirp_phase)
    else:
        chirp_wf = np.exp(1j * chirp_phase)

    return chirp_wf, window_support_pow2


def plot_wf_power(sig_wf1, sig_wf2, period_over_nyq, power1, power2, ybase2: bool = False):
    """
    Show the waveform and the averaged FFT over the whole record:
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(sig_wf1, label='input')
    ax1.plot(sig_wf2, '--', label='ifft')
    ax1.set_title('Synthetic Chirp')
    ax1.set_xlabel('N signal points')
    ax1.set_ylabel('Norm')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax2.plot(power1, label='fftw')
    ax2.plot(power2, '--', label='scipy fft')
    ax2.set_title(r'$|FFT|^2/||FFT||^2,  \omega_c = \pi/$' + str(period_over_nyq))
    ax2.set_xlabel('N frequency points to 2' + r'$\pi$')
    ax2.set_ylabel('Power')
    if ybase2:
        ax2.set_yscale('log', base=2)
    ax2.legend(loc='upper right')
    ax2.grid(True)


if __name__ == "__main__":
    order_nth = 12
    # Nyquist at pi
    period_over_nyquist = 8
    omega_center = np.pi/period_over_nyquist
    # Chirp index gamma, blueshift
    # gamma_index = 1/5  # For Q_s = Q_N
    gamma_index = 4
    gauss_taper = True

    sig_wf_complex, n_pow2 = chirp_sig(order=order_nth, omega=omega_center, gamma=gamma_index, gauss=gauss_taper)

    # Select waveform
    sig_wf_real = np.real(sig_wf_complex).astype(np.float32)
    sig_wf_imag = 0.0*np.copy(sig_wf_real)
    sig_wf = sig_wf_real + 1j * sig_wf_imag

    # Compute the signal power
    sig_power = np.abs(sig_wf)**2
    sig_power_sum_ave = np.sum(sig_power)/n_pow2
    sig_power_var = np.var(sig_wf_real)

    # The Forward FT does not have a 1/N scaling. When summing the coefficients, must divide by N^2

    # THE RFFT EXPECTS A REAL SIGNAL
    # Compute pyfftw forward rfft for real signals
    rfft_object = pyfftw.builders.rfft(sig_wf_real)
    # Object is constructed can be invoked at any time
    sig_rfft = rfft_object()
    # Compute scipy forward rfft
    sig_rfft_scipy = rfft(sig_wf_real)

    # THE FFT CAN TAKE A COMPLEX SIGNAL
    # Compute pyfftw forward fft
    fft_object = pyfftw.builders.fft(sig_wf)
    # Object is constructed can be invoked at any time
    sig_fft = fft_object()
    # Compute scipy forward fft
    sig_fft_scipy = fft(sig_wf)

    # Compute total power
    rfft_power = np.abs(sig_rfft) ** 2
    # MUST multiply RFFT power by 2; but use same number of signal points
    rfft_power_sum = 2*np.sum(rfft_power)
    rfft_power_sum_ave = rfft_power_sum/n_pow2**2

    rfft_power_sp = np.abs(sig_rfft_scipy) ** 2
    rfft_power_sum_sp = 2*np.sum(rfft_power_sp)
    rfft_power_sum_ave_sp = rfft_power_sum_sp/n_pow2**2

    # FFT power needs no adjustment
    fft_power = np.abs(sig_fft) ** 2
    fft_power_sum = np.sum(fft_power)
    fft_power_sum_ave = fft_power_sum/n_pow2**2

    fft_power_sp = np.abs(sig_fft_scipy) ** 2
    fft_power_sum_sp = np.sum(fft_power_sp)
    fft_power_sum_ave_sp = fft_power_sum_sp/n_pow2**2

    # The average variances match
    print(rfft_power_sum_ave, rfft_power_sum_ave_sp, fft_power_sum_ave, fft_power_sum_ave_sp, sig_power_sum_ave, sig_power_var)

    # Scaled spectrum = |FFT|^2/sum|FFT|^2|
    sig_spectrum = fft_power / fft_power_sum
    sig_spectrum_sp = fft_power_sp / fft_power_sum_sp
    # sig_spectrum_positive = sig_spectrum[0:n_pow2//2]

    # Compute ifft
    ifft_object = pyfftw.builders.ifft(sig_fft)
    sig_wf_ifft = ifft_object()

    plot_wf_power(sig_wf1=np.real(sig_wf), sig_wf2=np.real(sig_wf_ifft),
                  period_over_nyq=period_over_nyquist,
                  power1=sig_spectrum, power2=sig_spectrum_sp)
    plt.show()












