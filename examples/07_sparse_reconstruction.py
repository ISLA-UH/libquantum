"""
libquantum sparse reconstruction
Perform inverse CWT reconstruction as in Garces, 2020.
"""

import numpy as np
import matplotlib.pyplot as plt
from libquantum import synthetics as synth
from libquantum import blast_pulse as kaboom
from libquantum import utils, atoms, atoms_inverse
from libquantum.scales import EPSILON


def tfr_sparse(frequency, time, tfr):
    """
    
    :param frequency: TFR frequency
    :param time: TFR time
    :param tfr: TFR complex coefficients
    :return: 
    """

    # Grab the peak coefficient per frequency
    max_axis = 1
    # REAL
    coeff_real_argmax = np.argmax(np.abs(tfr.real), axis=max_axis)
    # IMAGINARY
    coeff_imag_argmax = np.argmax(np.abs(tfr.imag), axis=max_axis)

    # Build the sparse coefficient representation
    coeff_time_real_sparse = time[coeff_real_argmax]
    coeff_time_imag_sparse = time[coeff_imag_argmax]

    # "Fancy" indexing (it's a thing, see np.take)
    frequency_rows = np.arange(len(frequency))
    coeff_real_sparse = tfr.real[frequency_rows, coeff_real_argmax]
    coeff_imag_sparse = tfr.imag[frequency_rows, coeff_imag_argmax]

    return coeff_time_real_sparse, coeff_real_sparse, coeff_time_imag_sparse, coeff_imag_sparse


def tfr_sparse_sort(frequency_sparse, time_sparse, coeff_sparse):
    # Sort in descending order
    arg_coeff_sort = np.flip(np.argsort(np.abs(coeff_sparse)))
    coeff_sort = coeff_sparse[arg_coeff_sort]
    # Frequency of peak coefficient
    frequency_sort = frequency_sparse[arg_coeff_sort]
    # Time offset
    time_sort = time_sparse[arg_coeff_sort]
    
    return frequency_sort, time_sort, coeff_sort 


# Standalone plots
def plot_parameters():
    # Aspect ratio of 1920 x 1080 (1080p), 16:9
    scale = 1.25*1080/8
    figure_size_x = int(1920/scale)
    figure_size_y = int(1080/scale)
    text_size = int(2.9*1080/scale)
    return figure_size_x, figure_size_y, text_size


def plot_sparse(frequency_sparse, 
                coeff_time_real_sparse, 
                coeff_real_sparse, 
                coeff_time_imag_sparse, 
                coeff_imag_sparse):

    figure_size_x, figure_size_y, text_size = plot_parameters()
    fig = plt.figure(figsize=(figure_size_x, figure_size_y))
    plt.subplot(221)
    plt.plot(frequency_sparse,
             coeff_time_real_sparse, 'o')
    plt.grid(True)
    plt.title("Real Time offsets")

    plt.subplot(223)
    plt.plot(frequency_sparse,
             coeff_real_sparse, 'o')
    plt.grid(True)
    plt.title("Imag Time offsets")

    plt.subplot(222)
    plt.plot(frequency_sparse,
             coeff_time_imag_sparse, 'o')
    plt.grid(True)
    plt.title("Imag Time offsets")

    plt.subplot(224)
    plt.plot(frequency_sparse,
             coeff_imag_sparse, 'o')
    plt.grid(True)
    plt.title("Imag CW components")

    return fig


def plot_sorted(frequency_sort,
                coeff_time_sort,
                coeff_sort,
                title_prefix,
                coeff_number: int = 1):

    if coeff_number > len(frequency_sort):
        coeff_number = len(frequency_sort)-2  # Remove edges
    figure_size_x, figure_size_y, text_size = plot_parameters()
    fig = plt.figure(figsize=(figure_size_x, figure_size_y))
    plt.subplot(211)
    plt.plot(frequency_sort[0:coeff_number],
             coeff_time_sort[0:coeff_number], 'o')
    plt.grid(True)
    plt.title(title_prefix + " time offsets, M=" + str(coeff_number))

    plt.subplot(212)
    plt.plot(frequency_sort[0:coeff_number],
             coeff_sort[0:coeff_number], 'o')
    plt.grid(True)
    plt.title(title_prefix + " TFR coefficients")

    return fig


def plot_sparse_blast(synth_type, title, scaled_time, x_multiplier, y_max,
                      synth1, symbol1, label1,
                      synth2, symbol2, label2,
                      synth3, symbol3, label3):

    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/' + synth_type + '.png'
    fig = plt.figure(figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(111)
    ax1.plot(scaled_time, synth1, symbol1, label=label1)
    ax1.plot(scaled_time, synth2, symbol2, label=label2)
    ax1.plot(scaled_time, synth3, symbol3, label=label3)
    ax1.set_title(title, size=text_size)
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
    frequency_main_hz = 1.  # Nominal 5 Hz for 1 ton, after Kim et al. 2021
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

    # Build signal, no noise
    sig_gt = kaboom.gt_blast_period_center(time_center_s=time_shifted_s,
                                           pseudo_period_s=pseudo_period_s)
    # Add white noise
    # Variance computed from transient, stressing at bit_loss=1
    bit_loss = 6
    sig_noise = synth.white_noise_fbits(sig=sig_gt,
                                        std_bit_loss=bit_loss)
    gt_white = sig_gt + sig_noise

    # AA filter of signal with noise
    sig_n = synth.antialias_halfNyquist(synth=gt_white)  # With noise

    # Compute complex wavelet transform of real signal + noise
    cwtm, _, _, frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=sig_n,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_Nth)
    # Reconstruction coefficients
    _, reconstruct = \
        atoms_inverse.morlet2_reconstruct(band_order_Nth=order_Nth,
                                          scale_frequency_center_hz=frequency_hz,
                                          frequency_sample_rate_hz=frequency_sample_rate_hz)
    # Scaled wavelet coefficients
    f_x_cwtm = utils.d1tile_x_d2(d1=reconstruct,
                                 d2=cwtm.real)
    # Inverse to verify reconstruction (only works for real, returns Hilbert on imaginary
    sig_inv = np.sum(f_x_cwtm, 0)  # Reconstruction of signal with noise

    #############################
    # High level summary
    print('Order:', order_Nth)
    # Shape of cwtm
    print('CWT shape:', cwtm.shape)
    # Keep tabs on center frequency
    index_frequency_center = np.argmin(np.abs(frequency_hz-frequency_sig_hz))
    print(f"Closest center frequency to signal frequency is {frequency_hz[index_frequency_center]} Hz")

    # Extract sparse representation as a function of frequency
    time_real_sparse, cw_real_sparse, time_imag_sparse, cw_imag_sparse = \
        tfr_sparse(frequency=frequency_hz, time=time_s, tfr=cwtm)

    plot_sparse(frequency_sparse=frequency_hz,
                coeff_time_real_sparse=time_real_sparse,
                coeff_real_sparse=cw_real_sparse,
                coeff_time_imag_sparse=time_imag_sparse,
                coeff_imag_sparse=cw_imag_sparse)

    # Separate real and imaginary solutions
    # TODO: Return order, or zeroed out coefficients
    frequency_real_sort, time_real_sort, cw_real_sort = \
        tfr_sparse_sort(frequency_sparse=frequency_hz,
                        time_sparse=time_real_sparse,
                        coeff_sparse=cw_real_sparse)

    frequency_imag_sort, time_imag_sort, cw_imag_sort = \
        tfr_sparse_sort(frequency_sparse=frequency_hz,
                        time_sparse=time_imag_sparse,
                        coeff_sparse=cw_imag_sparse)

    # Energy and time
    # TODO: Resolve center frequency discrepancy, FFT spectral peak at 1 Hz
    coeff_scaled_energy_imag = cw_imag_sort**2/np.sum(cw_imag_sort**2)
    coeff_scaled_bits_imag = np.log2(np.abs(cw_imag_sort/cw_imag_sort[0]) + EPSILON)
    time_offset_re_max_imag = time_imag_sort - time_imag_sort[0]
    frequency_scaled_imag = frequency_imag_sort/frequency_imag_sort[0]

    # First coefficient arg (index 0) is the largest
    print("\nSparse rep of real TFR")
    print("Peak coefficient: ", cw_real_sort[0])
    print("Frequency of peak coefficient, Hz:", frequency_real_sort[0])
    print("Time of peak coefficient, s:", time_real_sort[0])
    print("\nSparse rep of imag TFR")
    print("Peak coefficient: ", cw_imag_sort[0])
    print("Frequency of peak coefficient, Hz:", frequency_imag_sort[0])
    print("Time of peak coefficient, s:", time_imag_sort[0])

    # Reset time so it is centered on max abs amplitude
    sig_time_real_s = time_s - time_real_sort[0]
    sig_time_imag_s = time_s - time_imag_sort[0]

    # Time offsets, zero for fundamental
    sig_time_real_sparse = time_real_sparse - time_real_sort[0]
    sig_time_imag_sparse = time_imag_sparse - time_imag_sort[0]

    print('Max absolute CWT coefficient:', np.max(np.abs(cwtm)))
    time_scaled = time_shifted_s*frequency_main_hz

    # Select number of modes
    j_real = 32
    j_imag = 32

    print("Number of bands:", len(frequency_hz))
    print("Number of real modes in the superposition:", j_real)
    print("Number of imag modes in the superposition:", j_imag)
    print()

    plot_sorted(frequency_sort=frequency_real_sort,
                coeff_time_sort=time_real_sort,
                coeff_sort=cw_real_sort,
                title_prefix="Real",
                coeff_number=j_real)

    plot_sorted(frequency_sort=frequency_imag_sort,
                coeff_time_sort=time_imag_sort,
                coeff_sort=cw_imag_sort,
                title_prefix="Imag",
                coeff_number=j_imag)

    # Plot the percent of the energy, per coefficient and cumulative
    plot_sorted(frequency_sort=frequency_scaled_imag,
                coeff_time_sort=time_offset_re_max_imag,
                coeff_sort=coeff_scaled_energy_imag,  # try coeff_scaled_bits_imag,
                title_prefix="Energy imag",
                coeff_number=j_imag)


    # Initialize modal superposition from largest contributions
    morl2_inv_real2 = np.zeros((len(frequency_hz), len(sig_n)))
    morl2_inv_imag2 = np.zeros((len(frequency_hz), len(sig_n)))

    for j_sorted_real in np.arange(j_real):
        morl2_inv_real2[j_sorted_real, :] = \
            atoms_inverse.inv_morlet2_real(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=time_real_sort[j_sorted_real],
                                           scale_frequency_center_hz=frequency_real_sort[j_sorted_real],
                                           cwt_amp_real=cw_real_sort[j_sorted_real],
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)

    for j_sorted_imag in np.arange(j_imag):
        morl2_inv_imag2[j_sorted_imag, :] = \
            atoms_inverse.inv_morlet2_imag(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=time_imag_sort[j_sorted_imag],
                                           scale_frequency_center_hz=frequency_imag_sort[j_sorted_imag],
                                           cwt_amp_imag=cw_imag_sort[j_sorted_imag],
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)

    # Complex waveform
    sig_superpose = np.sum(morl2_inv_real2, axis=0) + np.sum(morl2_inv_imag2, axis=0) * 1j

    # PLOTS
    fig_description = 'Reconstruction, all CWT coefficients'
    fig_title = fig_description
    plot_sparse_blast(fig_description, fig_title,
                      scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                      synth1=sig_gt, symbol1=".-", label1='Equation',
                      synth2=sig_n, symbol2="-", label2='AA_LP',
                      synth3=sig_inv, symbol3="-", label3='CWT Reconstruction')

    fig_description = 'CWT, top ' + str(j_real) + ' atoms'
    # fig_title = ''  # for publication
    fig_title = 'Reconstruction from real CWT'
    plot_sparse_blast(fig_description, fig_title,
                      scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                      synth1=sig_gt, symbol1=".-", label1='Equation',
                      synth2=sig_n, symbol2="-", label2='AA LP',
                      synth3=sig_superpose.real, symbol3="-", label3=fig_description)

    fig_description = 'CWT, top ' + str(j_imag) + ' atoms'
    fig_title = 'Reconstruction from imag CWT'
    plot_sparse_blast(fig_description, fig_title,
                      scaled_time=time_scaled, x_multiplier=2, y_max=1.4,
                      synth1=sig_gt, symbol1=".-", label1='Equation',
                      synth2=sig_n, symbol2="-", label2='AA LP',
                      synth3=sig_superpose.imag, symbol3="-", label3=fig_description)

    plt.show()
