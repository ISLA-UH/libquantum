"""
libquantum sparse reconstruction
Perform inverse CWT reconstruction as in Garces, 2020.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import synthetics as synth
from libquantum import blast_pulse as kaboom
from libquantum import utils, atoms, atoms_inverse
from libquantum.scales import EPSILON


# Standalone plots
def plot_parameters():
    # Aspect ratio of 1920 x 1080 (1080p), 16:9
    scale = 1.25*1080/8
    figure_size_x = int(1920/scale)
    figure_size_y = int(1080/scale)
    text_size = int(2.9*1080/scale)
    return figure_size_x, figure_size_y, text_size


def plot_sparse_blast(figure_number, synth_type, title, scaled_time, x_multiplier, y_max,
                      synth1, symbol1, label1,
                      synth2, symbol2, label2,
                      synth3, symbol3, label3):

    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/' + synth_type + '.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(111)
    ax1.plot(scaled_time, synth1, symbol1, label=label1)
    ax1.plot(scaled_time, synth2, symbol2, label=label2)
    ax1.plot(scaled_time, synth3, symbol3, label=label3)
    ax1.set_title(title, size = text_size)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_ylim(-y_max, y_max)
    ax1.set_xlim(-x_multiplier, x_multiplier)
    ax1.set_xlabel('Scaled time', size=text_size)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    return fig


if __name__ == "__main__":
    """
    Performs inverse CWT on GP pulse as in Garces (2021)
    """

    is_print_modes = False
    # Set Order
    order_Nth = 6

    # Target frequency
    frequency_main_hz = 4.  # Nominal 1 ton, after Kim et al. 2021
    pseudo_period_main_s = 1/frequency_main_hz
    # frequency_sample_rate_hz = 200.  # Recommend at least x16 target frequency
    frequency_sample_rate_hz = 32*frequency_main_hz  # Recommend at least x16 target frequency

    # Pulse frequency
    frequency_sig_hz = 1.*frequency_main_hz  # *np.sqrt(2) for 1/3 octave tests
    pseudo_period_s = 1/frequency_sig_hz

    # GT pulse relative to target (main) period
    # Number of cycles
    window_cycles = 64
    window_duration_s = window_cycles*pseudo_period_main_s
    # This will be the time to use 2^n. Note target frequency is used!
    time_points = int(window_duration_s*frequency_sample_rate_hz)
    time_s = np.arange(time_points)/frequency_sample_rate_hz
    time_half_s = np.max(time_s)/2.
    time_shifted_s = time_s - time_half_s
    time_scaled = time_shifted_s*frequency_main_hz

    # Build signal, no noise
    sig_gt = kaboom.gt_blast_period_center(time_center_s=time_shifted_s,
                                           pseudo_period_s=pseudo_period_s)
    sig_gt_hilbert = kaboom.gt_hilbert_blast_period_center(time_center_s=time_shifted_s,
                                                           pseudo_period_s=pseudo_period_s)
    sig_complex = sig_gt + sig_gt_hilbert*1j

    # Add white noise
    # Variance computed from transient
    bit_loss = 1
    sig_noise = synth.white_noise_fbits(sig=sig_gt,
                                        std_bit_loss=bit_loss)
    gt_white = sig_gt + sig_noise

    # AA filter of signal with noise
    noise = synth.antialias_halfNyquist(synth=sig_noise)
    sig_n = synth.antialias_halfNyquist(synth=gt_white)  # With noise

    # Compute complex wavelet transform of real signal + noise
    cwtm, _, _, frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=sig_n,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_Nth)

    cwtm_max_abs = np.max(np.abs(cwtm))
    # For noise
    cwtm_noise, _, _, _ = \
        atoms.cwt_chirp_from_sig(sig_wf=noise,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_Nth)

    frequency_scaled = frequency_hz/frequency_main_hz

    print('Order:', order_Nth)
    # Shape of cwtm
    print('CWT shape:', cwtm.shape)
    print('Max absolute CWT coefficient:', cwtm_max_abs)

    # Keep tabs on center frequency
    index_frequency_center = np.argmin(np.abs(frequency_hz-frequency_sig_hz))

    morl2_scale, reconstruct = \
        atoms_inverse.morlet2_reconstruct(band_order_Nth=order_Nth,
                                          scale_frequency_center_hz=frequency_hz,
                                          frequency_sample_rate_hz=frequency_sample_rate_hz)
    # Scaled wavelet coefficients
    f_x_cwtm = utils.d1tile_x_d2(d1=reconstruct,
                                 d2=cwtm)

    # Reference functions
    sig_theory = np.copy(sig_complex)  # No noise
    sig_hilbert = signal.hilbert(sig_n)  # With noise
    sig_inv = np.sum(f_x_cwtm, 0)  # Reconstruction of signal with noise

    # Initialize sparse coefficients
    m_cw_real_sparse = np.zeros(len(frequency_hz))
    m_cw_imag_sparse = np.zeros(len(frequency_hz))
    m_cw_time_real_sparse = np.zeros(len(frequency_hz))
    m_cw_time_imag_sparse = np.zeros(len(frequency_hz))

    # Initialize modal superposition from largest contributions
    morl2_inv_real2 = np.zeros((len(frequency_hz), len(sig_n)))
    morl2_inv_imag2 = np.zeros((len(frequency_hz), len(sig_n)))

    # TODO: First reassemble separately. Then reassemble with mix and match of real and imag
    # Change algorithm: sort by highest to lowest magnitude
    max_axis = 1
    # Grab the peak coefficient per frequency
    # REAL
    m_cw_real_maxabs = np.max(np.abs(cwtm.real), axis=max_axis)
    m_cw_real_argmax = np.argmax(np.abs(cwtm.real), axis=max_axis)
    # IMAGINARY
    m_cw_imag_maxabs = np.max(np.abs(cwtm.imag), axis=max_axis)
    m_cw_imag_argmax = np.argmax(np.abs(cwtm.imag), axis=max_axis)

    # Build the sparse coefficient representation
    # Construct a function (vectorize the method)
    for j_freq in np.arange(len(frequency_hz)):
        m_cw_real_sparse[j_freq] = cwtm[j_freq, m_cw_real_argmax[j_freq]].real
        m_cw_imag_sparse[j_freq] = cwtm[j_freq, m_cw_imag_argmax[j_freq]].imag
        m_cw_time_real_sparse[j_freq] = time_s[m_cw_real_argmax[j_freq]]
        m_cw_time_imag_sparse[j_freq] = time_s[m_cw_imag_argmax[j_freq]]

    # Sort in descending order
    arg_m_cw_real_sort = np.flip(np.argsort(m_cw_real_maxabs))
    arg_m_cw_imag_sort = np.flip(np.argsort(m_cw_imag_maxabs))

    # First coefficient arg (index 0) is the largest
    fundamental_wavelet_real_time_s = time_s[m_cw_real_argmax[0]]
    fundamental_wavelet_imag_time_s = time_s[m_cw_imag_argmax[0]]

    # Reset time so it is centered on max abs amplitude
    sig_time_real_s = time_s - fundamental_wavelet_real_time_s
    sig_time_imag_s = time_s - fundamental_wavelet_imag_time_s

    # print(arg_m_cw_real_sort)
    # print(m_cw_real_maxabs[arg_m_cw_real_sort])
    # print(arg_m_cw_imag_sort)
    # print(m_cw_imag_maxabs[arg_m_cw_imag_sort])

    # Select number of modes
    j_real = 32
    j_imag = 32

    print("Number of bands:", len(frequency_hz))
    print("Number of real modes in the superposition:", j_real)
    print("Number of imag modes in the superposition:", j_imag)
    print()

    for j_sorted_real in arg_m_cw_real_sort[0:j_real]:
        if is_print_modes:
            print("*Real CW Reconstruction")
            print(f"Frequency = {frequency_hz[j_sorted_real]} Hz")
            print(f"Scaled time offset = {time_scaled[m_cw_real_argmax[j_sorted_real]]}")
            print(f"CW coefficient abs amplitude = {m_cw_real_maxabs[j_sorted_real]}")
            print(f"CW coefficient amplitude = {m_cw_real_sparse[j_sorted_real]}")
        morl2_inv_real2[j_sorted_real, :] = \
            atoms_inverse.inv_morlet2_real(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=m_cw_time_real_sparse[j_sorted_real],
                                           scale_frequency_center_hz=frequency_hz[j_sorted_real],
                                           cwt_amp_real=m_cw_real_sparse[j_sorted_real],
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)

    print()
    for j_sorted_imag in arg_m_cw_imag_sort[0:j_imag]:
        if is_print_modes:
            print("*Imaginary CW Reconstruction")
            print(f"Frequency = {frequency_hz[j_sorted_imag]} Hz")
            print(f"Scaled time offset = {time_scaled[m_cw_imag_argmax[j_sorted_imag]]}")
            print(f"CW coefficient abs amplitude = {m_cw_imag_maxabs[j_sorted_imag]}")
            print(f"CW coefficient amplitude = {m_cw_imag_sparse[j_sorted_imag]}")
        morl2_inv_imag2[j_sorted_imag, :] = \
            atoms_inverse.inv_morlet2_imag(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=m_cw_time_imag_sparse[j_sorted_imag],
                                           scale_frequency_center_hz=frequency_hz[j_sorted_imag],
                                           cwt_amp_imag=m_cw_imag_sparse[j_sorted_imag],
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)

    # Complex waveform
    sig_superpose = np.sum(morl2_inv_real2, axis=0) + np.sum(morl2_inv_imag2, axis=0) * 1j

    # PLOTS
    fig_number = 1
    fig_description = 'Reconstruction, all CWT coefficients'
    fig_title = fig_description
    plot_sparse_blast(fig_number, fig_description, fig_title,
                 scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                 synth1=sig_theory.real, symbol1=".-", label1='Equation',
                 synth2=sig_hilbert.real, symbol2="-", label2='SciPy Hilbert',
                 synth3=sig_inv.real, symbol3="-", label3='CWT Reconstruction')

    fig_number += 1
    fig_description = 'CWT, top ' + str(j_real) + ' atoms'
    # fig_title = ''  # for publication
    fig_title = 'Reconstruction from real CWT'
    plot_sparse_blast(fig_number, fig_description, fig_title,
                 scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                 synth1=sig_theory.real, symbol1=".-", label1='Equation',
                 synth2=sig_hilbert.real, symbol2="-", label2='SciPy Hilbert',
                 synth3=sig_superpose.real, symbol3="-", label3=fig_description)

    fig_number += 1
    fig_description = 'CWT, top ' + str(j_imag) + ' atoms'
    fig_title = 'Reconstruction from imag CWT'
    plot_sparse_blast(fig_number, fig_description, fig_title,
                      scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                      synth1=sig_theory.real, symbol1=".-", label1='Equation',
                      synth2=sig_hilbert.real, symbol2="-", label2='SciPy Hilbert',
                      synth3=sig_superpose.imag, symbol3="-", label3=fig_description)

    plt.show()
