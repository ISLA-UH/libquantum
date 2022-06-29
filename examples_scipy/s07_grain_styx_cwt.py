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
    # Find FFT frequency near center
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    # FFT frequencies around the center
    frequency_cwt_fft_hz = frequency_fft_pos_hz[1:]  # [fft_index-3:fft_index+6]

    # Construct single wavelet from center FFT frequency
    # TODO: Option, superpose the wavelets and reconstruct
    mic_sig_complex, time_s, _, _, _ = \
        atoms_styx.wavelet_centered_4cwt(band_order_Nth=order_number_input,
                                         duration_points=time_nd,
                                         scale_frequency_center_hz=frequency_center_fft_hz,
                                         frequency_sample_rate_hz=frequency_sample_rate_hz,
                                         dictionary_type="unit")

    mic_sig_real = np.real(mic_sig_complex)
    mic_sig_imag = np.imag(mic_sig_complex)

    sig_wf = np.copy(mic_sig_real)
    # Convert to a 2d matrix
    sig_wf_2d = np.tile(sig_wf, (len(frequency_cwt_fft_hz), 1))
    # Take signal fft
    sig_fft_2d = np.fft.fft(sig_wf_2d)
    # # Convert to a 2d matrix
    # sig_fft_2d = np.tile(sig_fft, (len(frequency_cwt_fft_hz), 1))

    # Construct CWT wavelets from center FFT frequencies near signal
    cw_complex, cw_time_s, scale, omega, amp = \
        atoms_styx.wavelet_centered_4cwt(band_order_Nth=order_number_input,
                                         duration_points=time_nd,
                                         scale_frequency_center_hz=frequency_cwt_fft_hz,
                                         frequency_sample_rate_hz=frequency_sample_rate_hz,
                                         dictionary_type="norm")

    # Flip the time axis (-t)
    cw_complex_fliplr = np.fliplr(cw_complex)
    # Test and verify this is correct - compare to signal.cwt
    atom_fft = np.fft.fft(cw_complex)
    cwt_raw = np.fft.ifft(sig_fft_2d*np.conj(atom_fft))
    cwt_fft = np.append(cwt_raw[:, time_nd//2:], cwt_raw[:, 0:time_nd//2], axis=-1)
    cwt_conv = np.empty((len(frequency_cwt_fft_hz), time_nd), dtype=np.complex128)
    for j in range(len(frequency_cwt_fft_hz)):
        cwt_conv[j, :] = signal.convolve(sig_wf, np.conj(cw_complex_fliplr[j, :]), mode='same')
    cwt_conv_fft = signal.fftconvolve(sig_wf_2d, np.conj(cw_complex_fliplr), mode='same', axes=-1)
    # cwt_conv2d = signal.convolve2d(sig_wf_2d, np.conj(cw_complex)[:, ::-1], mode='full')

    print(sig_fft_2d.shape)
    print(atom_fft.shape)
    print(cwt_raw.shape)
    print(cwt_fft.shape)
    print(cwt_conv.shape)
    # print(cwt_conv2d.shape)

    plt.matshow(np.abs(cwt_conv_fft))
    plt.show()
    exit()

    fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, np.sig_wf)
    ax1.set_title('Real Atom')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax1.grid(True)
    ax2.plot(time_s, np.imag(mic_sig_complex))
    ax2.set_title('Imag Atom')
    ax2.set_xlabel('Time, s')
    ax2.set_ylabel('Norm')
    ax2.grid(True)

    # Show the waveform(s)
    fig2, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
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