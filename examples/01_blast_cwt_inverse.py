"""
libquantum example 1: 01_blast_cwt_inverse.py
Perform inverse CWT reconstruction as in Garces, 2020.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import synthetics as synth
from libquantum import blast_pulse as kaboom
from libquantum import utils, atoms, atoms_inverse
from libquantum.scales import EPSILON


def energy_pdf_entropy(cwcoeff_complex):
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficients (not power), takes the square
    energy_pdf_real = cwcoeff_complex.real**2/np.sum(np.abs(cwcoeff_complex*np.conj(cwcoeff_complex)))
    energy_pdf_imag = cwcoeff_complex.imag**2/np.sum(np.abs(cwcoeff_complex*np.conj(cwcoeff_complex)))

    print('PDF Energy distribution: real, imaginary, total:',
          np.sum(energy_pdf_real), np.sum(energy_pdf_imag),
          np.sum(energy_pdf_real) + np.sum(energy_pdf_imag))

    energy_pdf = energy_pdf_real + energy_pdf_imag*1j

    # Log energy entropy (LEE) = log(p)
    entropy_LEE = 0.5*np.log2(energy_pdf.real + EPSILON) + \
                  0.5*np.log2(energy_pdf.imag + EPSILON)*1j

    # Shannon Entropy (SE) = -p*log(p)
    entropy_SE = energy_pdf.real*entropy_LEE.real + \
                 energy_pdf.imag*entropy_LEE.imag*1j
    entropy_SE *= -2.

    return energy_pdf, entropy_LEE, entropy_SE


def snr_pdf_entropy(snr_complex):
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes squared (power) inputs

    snr_total = np.sum(np.sqrt(np.abs(snr_complex*np.conj(snr_complex))))
    snr_pdf_real = snr_complex.real/snr_total
    snr_pdf_imag = snr_complex.imag/snr_total

    print('PDF SNR distribution: real, imaginary, total: Varies!',
          np.sum(snr_pdf_real), np.sum(snr_pdf_imag),
          np.sum(snr_pdf_real) + np.sum(snr_pdf_imag))

    energy_pdf = snr_pdf_real + snr_pdf_imag*1j

    # Log energy entropy (LEE) = log(p)
    entropy_LEE = 0.5*np.log2(energy_pdf.real + EPSILON) + \
                  0.5*np.log2(energy_pdf.imag + EPSILON)*1j

    # Shannon Entropy (SE) = -p*log(p)
    entropy_SE = energy_pdf.real*entropy_LEE.real + \
                 energy_pdf.imag*entropy_LEE.imag*1j
    entropy_SE *= -2.

    return energy_pdf, entropy_LEE, entropy_SE


def noise_snr(cwcoeff, sort_array_complex, sort_threshold):

    # Takes the lowest noise_coeff of the cwcoeff coefficients
    noise_power_real = np.zeros(cwcoeff.shape[0])
    noise_power_imag = np.zeros(cwcoeff.shape[0])
    for j in np.arange(cwcoeff.shape[0]):
        index_real = np.where(sort_array_complex[j, :].real < sort_threshold.real)
        noise_power_real[j] = np.mean(cwcoeff[j, index_real].real**2)
        index_imag = np.where(sort_array_complex[j, :].imag < sort_threshold.imag)
        noise_power_imag[j] = np.mean(cwcoeff[j, index_imag].imag**2)
    # Mean across the whole band to avoid penalizing the main peak
    noise_power = np.mean(noise_power_real)*np.ones(cwcoeff.shape) + \
                  np.mean(noise_power_imag)*np.ones(cwcoeff.shape)*1j
    snr_power = cwcoeff.real**2/noise_power.real + cwcoeff.imag**2/noise_power.imag*1j

    return snr_power, noise_power


# Standalone plots
def plot_parameters():
    # Aspect ratio of 1920 x 1080 (1080p), 16:9
    scale = 1.25*1080/8
    figure_size_x = int(1920/scale)
    figure_size_y = int(1080/scale)
    text_size = int(2.9*1080/scale)
    return figure_size_x, figure_size_y, text_size


def plot_complex(figure_number, synth_type, title, scaled_time, x_multiplier, y_max,
                 synth1, symbol1, label1,
                 synth2, symbol2, label2,
                 synth3, symbol3, label3):

    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/' + synth_type + '.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(121)
    ax1.plot(scaled_time, synth1.real, symbol1, label=label1)
    ax1.plot(scaled_time, synth2.real, symbol2, label=label2)
    ax1.plot(scaled_time, synth3.real, symbol3, label=label3)
    ax1.set_title(title, size = text_size)
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax1.set_ylim(-y_max, y_max)
    ax1.set_xlim(-x_multiplier, x_multiplier)
    ax1.set_xlabel('Scaled time', size=text_size)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)
    ax2 = plt.subplot(122)
    ax2.plot(scaled_time, synth1.imag, symbol1, label=label1)
    ax2.plot(scaled_time, synth2.imag, symbol2, label=label2)
    ax2.plot(scaled_time, synth3.imag, symbol3, label=label3)
    ax2.legend(loc='upper right')
    ax2.grid(True)
    ax2.set_ylim(-y_max, y_max)
    ax2.set_xlim(-x_multiplier, x_multiplier)
    ax2.set_xlabel('Scaled time', size=text_size)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)
    fig.tight_layout()

    return fig


def plot_wiggles_complex_label(figure_number, xarray, wf_array, wf_label, xlim_max,
                               y0_color, y0_label, y_color):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))

    wiggle_num = wf_array.shape[0]
    offset_scaling = 1./4.
    wiggle_offset = np.arange(0, wiggle_num)*offset_scaling
    wiggle_yticks = wiggle_offset
    wiggle_yticklabel = wf_label

    ax1 = plt.subplot(121)
    ax1.set_yticks(wiggle_yticks)
    ax1.set_yticklabels('{:.2f}'.format(a) for a in wiggle_yticklabel)
    ax1.set_ylim(wiggle_offset[0]-2*offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax1.set_xlim(-xlim_max, xlim_max)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)

    for j in np.arange(wiggle_num):
        if j == 0:
            ax1.plot(xarray, wf_array[j, :].real+wiggle_offset[j], color=y0_color, label=y0_label+', Real')
        else:
            ax1.plot(xarray, wf_array[j, :].real+wiggle_offset[j], color=y_color)
    ax1.grid(True)
    ax1.set_ylabel('Scaled frequency', size=text_size)
    ax1.set_xlabel('Scaled time', size=text_size)
    ax1.legend(loc='lower right')

    ax2 = plt.subplot(122)
    ax2.set_yticks(wiggle_yticks)
    ax2.set_yticklabels([])
    ax2.set_ylim(wiggle_offset[0]-2*offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax2.set_xlim(-xlim_max, xlim_max)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)

    for j in np.arange(wiggle_num):
        if j == 0:
            ax2.plot(xarray, wf_array[j, :].imag+wiggle_offset[j], color=y0_color, label=y0_label+', Imag')
        else:
            ax2.plot(xarray, wf_array[j, :].imag+wiggle_offset[j], color=y_color)
    ax2.grid(True)
    ax2.set_xlabel('Scaled time', size=text_size)
    ax2.legend(loc='lower right')
    fig.tight_layout()


if __name__ == "__main__":
    """
    Performs inverse CWT on GP pulse as in Garces (2021)
    """

    # Set Order
    order_Nth = 3

    frequency_sample_rate_hz = 200.
    # Target frequency
    frequency_main_hz = 5  # Nominal 1 ton, after Kim et al. 2021
    pseudo_period_main_s = 1/frequency_main_hz
    # Pulse frequency
    frequency_sig_hz = 1.*frequency_main_hz  # *np.sqrt(2)
    pseudo_period_s = 1/frequency_sig_hz

    # GT pulse
    # Number of cycles
    window_cycles = 64
    window_duration_s = window_cycles*pseudo_period_main_s
    # This will be the time to use 2^n
    time_points = int(window_duration_s*frequency_sample_rate_hz)
    time_s = np.arange(time_points)/frequency_sample_rate_hz
    time_half_s = np.max(time_s)/2.
    time_shifted_s = time_s - time_half_s
    time_scaled = time_shifted_s*frequency_main_hz

    # Select signal
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
    gt_white_hilbert = sig_gt_hilbert + sig_noise
    # AA filter
    noise = synth.antialias_halfNyquist(synth=sig_noise)
    sig_n = synth.antialias_halfNyquist(synth=gt_white)
    sig_n_hilbert = synth.antialias_halfNyquist(synth=gt_white_hilbert)
    # Analytic record
    sig_n_complex = sig_n + sig_n_hilbert*1j

    # Compute complex wavelet transform
    cwtm, _, _, frequency_hz = \
        atoms.cwt_chirp_from_sig(sig_wf=sig_n,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_Nth)
    # For noise
    cwtm_noise, _, _, _ = \
        atoms.cwt_chirp_from_sig(sig_wf=noise,
                                 frequency_sample_rate_hz=frequency_sample_rate_hz,
                                 band_order_Nth=order_Nth)

    frequency_scaled = frequency_hz/frequency_main_hz

    print('Order:', order_Nth)
    # Shape of cwtm
    print('CWT shape:', cwtm.shape)
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
    sig_inv = np.sum(f_x_cwtm, 0)
    sig_hilbert = signal.hilbert(sig_n)
    sig_theory = 1.*sig_complex
    noise_hilbert = signal.hilbert(noise)

    # Entropy
    energy_pdf, entropy_LEE, entropy_SE = energy_pdf_entropy(cwcoeff_complex=cwtm)

    # Compute SNR from noise directly - still oscillates
    # Creates instability due to variability, specifically at low frequencies
    # Real and imaginary part of noise are the same
    # Fast approach, less vulnerable
    noise_power = np.mean(cwtm_noise.real**2) * np.ones(len(frequency_hz)) * (1 + 1j)

    # Flatten the noise
    cwtm_noise_power_tile = utils.just_tile(array1d_in=noise_power.real, shape_out=cwtm_noise.shape) + \
                            utils.just_tile(array1d_in=noise_power.imag, shape_out=cwtm_noise.shape)*1j

    snr = cwtm.real**2/cwtm_noise_power_tile.real + \
          cwtm.imag**2/cwtm_noise_power_tile.imag*1j

    # Build a PDF and entropy of the traditional SNR
    snr_pdf, snr_LEE, snr_SE = snr_pdf_entropy(snr_complex=snr)

    # Modal superposition from largest contributions
    morl2_inv_real2 = np.zeros((len(frequency_hz), len(sig_n)))
    morl2_inv_imag2 = np.zeros((len(frequency_hz), len(sig_n)))

    super_array = np.copy(snr_SE.real) + 1j*np.copy(snr_SE.imag)
    super_array_max = np.max(super_array)
    cutoff = 0*1./2**6
    mode_counter = 0
    print('Max:', super_array_max)
    for j in np.arange(len(frequency_hz)):
        m_cw_real = np.argmax(super_array[j, :].real)
        m_cw_imag = np.argmax(super_array[j, :].imag)
        f_j_hz = frequency_hz[j]
        t_m_off_s = time_s[m_cw_real] + \
                    time_s[m_cw_imag]*1j

        condition1 = super_array[j, m_cw_real] >= cutoff*super_array_max.real
        condition2 = super_array[j, m_cw_imag] >= cutoff*super_array_max.imag
        if condition1 and condition2:
            c_m_amp = cwtm[j, m_cw_real].real + 1j*cwtm[j, m_cw_imag].imag
            mode_counter += 1
        else:
            c_m_amp = 0*(1 + 1j)

        morl2_inv_real2[j, :] = \
            atoms_inverse.inv_morlet2_real(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=t_m_off_s.real,
                                           scale_frequency_center_hz=f_j_hz,
                                           cwt_amp_real=c_m_amp.real,
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)
        # must use cosine
        morl2_inv_imag2[j, :] = \
            atoms_inverse.inv_morlet2_real(band_order_Nth=order_Nth,
                                           time_s=time_s,
                                           offset_time_s=t_m_off_s.imag,
                                           scale_frequency_center_hz=f_j_hz,
                                           cwt_amp_real=c_m_amp.imag,
                                           frequency_sample_rate_hz=frequency_sample_rate_hz)

    print("Number of bands:", len(frequency_hz))
    print("Number of modes in the superposition:", mode_counter)
    sig_superpose = np.sum(morl2_inv_real2, axis=0) + np.sum(morl2_inv_imag2, axis=0) * 1j

    # PLOTS
    fig_number = 1
    fig_description = 'Reconstruction, Hilbert comparisons'
    fig_title = ''  # for publication
    plot_complex(fig_number, fig_description, fig_title,
                 scaled_time=time_scaled, x_multiplier=1.0, y_max=1,
                 synth1=sig_theory, symbol1=".-", label1='Equation',
                 synth2=sig_hilbert, symbol2="-", label2='SciPy Hilbert',
                 synth3=sig_inv, symbol3="-", label3='CWT Reconstruction')

    fig_number += 1
    fig_description = 'CWT, top ' + str(mode_counter) + ' atoms'
    fig_title = ''  # for publication
    plot_complex(fig_number, fig_description, fig_title,
                 scaled_time=time_scaled, x_multiplier=1.0, y_max=1,
                 synth1=sig_theory, symbol1=".-", label1='Equation',
                 synth2=sig_hilbert, symbol2="-", label2='SciPy Hilbert',
                 synth3=sig_superpose, symbol3="-", label3=fig_description)

    fig_number += 1
    # 'Wiggle plots, scaled coefficients'
    sig_arr = sig_hilbert.reshape(1, sig_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, f_x_cwtm))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Input", y_color="black")

    fig_number += 1
    # 'Wiggle plots, unscaled sig coefficients'
    sig_arr = sig_hilbert.reshape(1, sig_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, cwtm))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Input", y_color="C0")

    fig_number += 1
    # 'Wiggle plots, unscaled noise coefficients'
    sig_arr = noise_hilbert.reshape(1, noise_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, cwtm_noise))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Noise", y_color="C0")

    fig_number += 1
    # 'Wiggle plots, scaled entropy'
    xY = 3*order_Nth
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=xY*entropy_SE, wf_label=frequency_scaled, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Shannon Entropy", y_color="C0")

    fig_number += 1
    # 'Wiggle plots, SNR entropy, noise re cw'
    xY = 3*order_Nth
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=xY*snr_SE, wf_label=frequency_scaled, xlim_max=order_Nth,
                               y0_color="C0", y0_label="SNR RbR", y_color="C0")

    plt.show()
