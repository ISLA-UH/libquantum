"""
libquantum example: s07_grain_tfr

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import utils, synthetics, atoms_styx
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq
from libquantum.styx import tfr_stx_fft

print(__doc__)
EVENT_NAME = 'grain test'
station_id_str = 'synth'
# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1

frequency_center_hz = 60
frequency_sample_rate_hz = 800
order_number_input = 24
time_nd = 2**11
time_fft_nd = 2**7

if __name__ == "__main__":
    """
    Compute the spectrogram over sliding windows.
    The Welch method is equivalent to averaging the spectrogram over the columns.
    """

    frequency_resolution_fft_hz = frequency_sample_rate_hz/time_fft_nd
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz-frequency_center_hz))
    # frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    #
    # Construct wavelets around the center
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index-1:fft_index+2]

    mic_sig_complex, time_s, scale, omega, amp = \
        atoms_styx.wavelet_centered_4cwt(band_order_Nth=order_number_input,
                                         duration_points=time_nd,
                                         scale_frequency_center_hz=frequency_center_fft_hz,
                                         frequency_sample_rate_hz=frequency_sample_rate_hz,
                                         dictionary_type="unit")

    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    # Computed Variance; divides by the number of points
    mic_sig_real_var = np.var(mic_sig_real, 1)
    mic_sig_imag_var = np.var(mic_sig_imag, 1)

    # Computed Variance * Number of Samples ~ integral. The dictionary type = "norm" returns 1/2.
    mic_sig_real_integral = np.var(mic_sig_real)*len(mic_sig_real)
    mic_sig_imag_integral = np.var(mic_sig_imag)*len(mic_sig_real)

    # Theoretical variance
    amp_f = amp[:, 0]
    scale_f = scale[:, 0]
    omega_f = omega[:, 0]
    mic_sig_real_var_nominal = amp_f**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale_f * \
                               (1 + np.exp(-(scale_f*omega_f)**2))
    mic_sig_imag_var_nominal = amp_f**2/len(time_s) * 0.5*np.sqrt(np.pi)*scale_f * \
                               (1 - np.exp(-(scale_f*omega_f)**2))

    print('mic_sig_real_variance:', mic_sig_real_var)
    print('real_variance_nominal:', mic_sig_real_var_nominal)
    print('mic_sig_imag_variance:', mic_sig_imag_var)
    print('imag_variance_nominal:', mic_sig_imag_var_nominal)

    # Show the waveform(s)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))

    if np.isscalar(frequency_center_fft_hz):
        ax1.plot(time_s, np.real(mic_sig_complex))
        ax1.set_title('Real Atom')
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('Norm')
        ax1.grid(True)
        ax2.plot(time_s, np.imag(mic_sig_complex))
        ax2.set_title('Imag Atom')
        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('Norm')
        ax2.grid(True)
    else:
        for j in np.arange(len(frequency_center_fft_hz)):
            ax1.plot(time_s, np.real(mic_sig_complex)[j, :])
            ax2.plot(time_s, np.imag(mic_sig_complex)[j, :])
        ax1.set_title('Real Atom')
        ax1.set_xlabel('Time, s')
        ax1.set_ylabel('Norm')
        ax2.set_title('Imag Atom')
        ax2.set_xlabel('Time, s')
        ax2.set_ylabel('Norm')
        ax1.grid(True)
        ax2.grid(True)
    plt.show()