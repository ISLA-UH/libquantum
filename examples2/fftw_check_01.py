import pyfftw
import numpy as np
import matplotlib.pyplot as plt


def chirp_sig(order: float, omega: float, gamma: float, gauss: bool=True):
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


if __name__ == "__main__":
    order_nth = 12
    # Nyquist at pi
    omega_center = np.pi/16
    # Chirp index gamma, blueshift
    gamma_index = 2

    sig_wf, n_pow2 = chirp_sig(order=order_nth, omega=omega_center, gamma=0, gauss=False)

    # Compute rfft
    fft_object = pyfftw.builders.fft(np.real(sig_wf))
    # Object is constructed can be invoked at any time
    sig_fft = fft_object()

    # Compute total power
    sig_power = np.real(np.sum(sig_fft*sig_fft.conjugate()))
    sig_spectrum = np.abs(sig_fft)**2
    sig_spectrum /= sig_power

    # Compute irfft
    ifft_object = pyfftw.builders.ifft(sig_fft)
    sig_wf_ifft = ifft_object()

    plt.figure()
    plt.plot(np.real(sig_wf))
    plt.plot(np.real(sig_wf_ifft))
    plt.figure()
    plt.semilogx(sig_spectrum[0:n_pow2//2])

    plt.show()












