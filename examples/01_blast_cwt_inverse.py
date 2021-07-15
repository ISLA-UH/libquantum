import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import milton.hellborne.hellborne_wavelet as hell
import scipy.signal as signal
import analysis_beta.synthetics.synth_beta as synth
# Placing it here for later copypaste into main
EPSILON = np.finfo(np.float128).eps


def just_tile(array1d_in: np.ndarray, shape_out) -> np.ndarray:

    if len(shape_out) == 1:
        tiled_matrix = np.tile(array1d_in, (shape_out[0]))
    elif len(shape_out) == 2:
        tiled_matrix = np.tile(array1d_in, (shape_out[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(array1d_in.shape)))

    return tiled_matrix

def white_noise(synth, snr_bits):
    synth_max = np.max(np.abs(synth))
    synth_std = np.std(synth)
    # TODO: selection criteria for SNR
    synth_snr_bits = 2.**snr_bits
    std_from_bits = synth_std/synth_snr_bits
    # White noise, zero mean
    synth_noise = np.random.normal(0, std_from_bits, size=synth.size)
    # TODO: Error correction for bandedges
    return synth_noise

def cwt_complex(sig, frequency_center_hz, sample_rate_hz, order_Nth):

    # Pick wavelet order and 2^n points
    order_bandedge = 2 ** (1. / 2. / order_Nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1. / order_bandedge
    quality_factor_Q = 1./order_scaled_bandwidth  # Exact for Nth octave bands
    cycles_M = quality_factor_Q*2*np.sqrt(np.log(2))
    sig_duration_s = len(sig)/sample_rate_hz

    # TODO: Turn into a function, look at slice
    n_min = np.ceil(order_Nth*np.log2(2/(frequency_center_hz*sig_duration_s)))
    n_max = np.floor(order_Nth*np.log2(sample_rate_hz/(2*frequency_center_hz)))
    n = np.arange(n_min, n_max+1)
    frequency_hz = frequency_center_hz*2**(n/order_Nth)
    frequency_hz_plot = np.append(frequency_hz*2**(-1/(2*order_Nth)), frequency_hz[-1]*2**(1/(2*order_Nth)))
    # widths is scales s in scipy.morlet2
    scales = sample_rate_hz*cycles_M/(2*np.pi*frequency_hz)
    w = cycles_M  # Default is 5 for morlet2
    # Signal Scipy
    cwtm = signal.cwt(sig, signal.morlet2, scales, w=w)

    return cwtm, frequency_hz, frequency_hz_plot, sample_rate_hz, \
           order_Nth, cycles_M, quality_factor_Q


def d1tile_x_d2(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    shape_out = d2.shape
    # create array of repeated values of PSD with dimensions that match those of energy array
    if len(shape_out) == 1:
        d1_matrix = np.tile(d1, (shape_out[0]))
    elif len(shape_out) == 2:
        d1_matrix = np.tile(d1, (shape_out[1], 1)).T
    else:
        raise TypeError('Cannot handle an array of shape {}.'.format(str(d1.shape)))
    d1_x_d2 = d1_matrix*d2
    return d1_x_d2


def energy_pdf_entropy(cwcoeff_complex):
    # Evaluate Log energy entropy (LEE) = log(p) and Shannon Entropy (SE) = -p*log(p)
    # Assumes linear spectral coefficien ts (not power), takes the square
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
    # Mean across the whole band to avoid punishing the main peak
    noise_power = np.mean(noise_power_real)*np.ones(cwcoeff.shape) + \
                  np.mean(noise_power_imag)*np.ones(cwcoeff.shape)*1j
    snr_power = cwcoeff.real**2/noise_power.real + cwcoeff.imag**2/noise_power.imag*1j

    return snr_power, noise_power

def hell_M_from_N(band_order_Nth):

    order_bandedge = 2 ** (1. / 2. / band_order_Nth)  # kN in Garces 2013
    order_scaled_bandwidth = order_bandedge - 1. / order_bandedge
    quality_factor_Q = 1./order_scaled_bandwidth  # Exact for Nth octave bands
    cycles_M = quality_factor_Q*2*np.sqrt(np.log(2))  # Exact, from -3dB points
    return cycles_M, quality_factor_Q

def morl2_reconstruct(band_order_Nth, frequency_center_hz, sample_rate_hz):

    cycles_M, quality_factor_Q = hell_M_from_N(band_order_Nth)
    morl2_scale = cycles_M*sample_rate_hz/frequency_center_hz/(2. * np.pi)
    reconstruct = np.pi**0.25/2/np.sqrt(morl2_scale)
    return morl2_scale, reconstruct

def inv_morl2_prep(band_order_Nth, time_s, offset_time_s, frequency_center_hz, sample_rate_hz):

    cycles_M, quality_factor_Q = hell_M_from_N(band_order_Nth)
    morl2_scale, reconstruct = morl2_reconstruct(order_Nth, frequency_center_hz, sample_rate_hz)
    xtime_shifted = sample_rate_hz*(time_s-offset_time_s)

    return xtime_shifted, morl2_scale, cycles_M, reconstruct

def inv_morl2_real(order_Nth, time_s, offset_time_s, frequency_center_hz, cwt_amp_real, sample_rate_hz):

    xtime_shifted, xscale, cycles_M, rescale = \
        inv_morl2_prep(order_Nth, time_s, offset_time_s, frequency_center_hz, sample_rate_hz)
    wavelet_gauss =  np.exp(-0.5 * (xtime_shifted / xscale) ** 2)
    wavelet_gabor_real = wavelet_gauss * np.cos(cycles_M*(xtime_shifted / xscale))

    # Rescale to Morlet wavelet and take the conjugate for imag
    morl2_inv_real = cwt_amp_real*wavelet_gabor_real
    morl2_inv_real *= rescale

    return morl2_inv_real


def inv_morl2_imag(order_Nth, time_s, offset_time_s, frequency_center_hz, cwt_amp_imag, sample_rate_hz):
    # TODO: Explain why pi/2 shift has to be removed!
    xtime_shifted, xscale, cycles_M, rescale = \
        inv_morl2_prep(order_Nth, time_s, offset_time_s, frequency_center_hz, sample_rate_hz)

    wavelet_gauss =  np.exp(-0.5 * (xtime_shifted / xscale) ** 2)
    wavelet_gabor_imag = wavelet_gauss * np.sin(cycles_M*(xtime_shifted / xscale))

    # Rescale to Morlet wavelet and take the conjugate for imag
    morl2_inv_imag = -cwt_amp_imag*wavelet_gabor_imag
    morl2_inv_imag *= rescale

    return morl2_inv_imag


# Standalone plots
def plot_parameters():
    # Aspect ratio of 1920 x 1080 (1080p), 16:9
    # scale = 1/3 => 640 x 360 (360p)
    # scale = 2/3 =>  1280 x 720 (720p)
    # scale = 4/3 =>  2560 x 1440 (1440p)
    # scale = 2 => 3840 x 2160 (2160p)
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
    figure_name = './figures/'+ synth_type +'.png'
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
    # ax2.set_xlabel(r'$t/t_{main}$', size = text_size)
    ax2.set_xlabel('Scaled time', size=text_size)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)
    fig.tight_layout()
    # if save_fig:
    #     fig.savefig(figure_name, dpi=300)
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
    # print(wiggle_yticklabel)

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
    # ax2.set_yticklabels(wiggle_yticklabel)
    # ax2.set_yticklabels('{:.2f}'.format(a) for a in wiggle_yticklabel)
    ax2.set_ylim(wiggle_offset[0]-2*offset_scaling, wiggle_offset[-1]+offset_scaling)
    ax2.set_xlim(-xlim_max, xlim_max)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)

    for j in np.arange(wiggle_num):
        if j==0:
            ax2.plot(xarray, wf_array[j, :].imag+wiggle_offset[j], color=y0_color, label=y0_label+', Imag')
        else:
            ax2.plot(xarray, wf_array[j, :].imag+wiggle_offset[j], color=y_color)
    ax2.grid(True)
    ax2.set_xlabel('Scaled time', size=text_size)
    ax2.legend(loc='lower right')
    fig.tight_layout()


if __name__ == "__main__":
    # TODO MAG: change description to explicitly explain what the code does
    """
    Do it all, clean it up, press-ready figures
    """
    ###### BEGIN #######
    order_Nth_main = 3

    frequency_sample_rate_hz = 200.
    # frequency_center_hz = frequency_sample_rate_hz/64.
    # Target frequency
    frequency_main_hz = 6.3
    pseudo_period_main_s = 1/frequency_main_hz
    # Pulse frequency
    frequency_sig_hz = 1.*frequency_main_hz  # *np.sqrt(2)
    pseudo_period_s = 1/frequency_sig_hz

    # GT pulse
    # Number of cycles
    window_cycles = 16
    window_duration_s = window_cycles*pseudo_period_main_s
    # This will be the time to use 2^n
    time_points = int(window_duration_s*frequency_sample_rate_hz)
    time_s = np.arange(time_points)/frequency_sample_rate_hz
    time_half_s = np.max(time_s)/2.
    time_shifted_s = time_s - time_half_s
    time_scaled = time_shifted_s*frequency_main_hz

    # Select signal
    sig_gt = hell.gt_blast_period_center(time_shifted_s, pseudo_period_s)
    sig_gt_hilbert = hell.gt_hilbert_blast_period_center(time_shifted_s, pseudo_period_s)
    sig_complex = sig_gt + sig_gt_hilbert*1j
    # Add white noise
    # Variance computed from transient
    bit_loss = 1
    sig_noise = white_noise(sig_gt, bit_loss)
    gt_white = sig_gt + sig_noise
    gt_white_hilbert = sig_gt_hilbert + sig_noise
    # AA filter
    noise = synth.antialias_halfNyquist(sig_noise)
    sig_n = synth.antialias_halfNyquist(gt_white)
    sig_n_hilbert = synth.antialias_halfNyquist(gt_white_hilbert)
    # Analytic record
    sig_n_complex = sig_n + sig_n_hilbert*1j

    # Compute complex wavelet transform
    cwtm, frequency_hz, frequency_hz_plot, sample_rate_hz, order_Nth, cycles_M, quality_factor_Q = \
        cwt_complex(sig=sig_n, frequency_center_hz=frequency_sig_hz,
                    sample_rate_hz=frequency_sample_rate_hz, order_Nth=order_Nth_main)
    # For noise
    cwtm_noise, fn, sn, sn, Nn, Mn, Qn = \
        cwt_complex(sig=noise, frequency_center_hz=frequency_sig_hz,
                    sample_rate_hz=frequency_sample_rate_hz, order_Nth=order_Nth_main)

    frequency_scaled = frequency_hz/frequency_main_hz

    print('Order:', order_Nth)
    # Shape of cwtm
    print('CWT shape:', cwtm.shape)
    # Keep tabs on center frequency
    index_frequency_center = np.argmin(np.abs(frequency_hz-frequency_sig_hz))
    # Print tuning frequency
    # print(frequency_hz[index_frequency_center], '{:0.2f}'.format)

    # TODO: One function return
    morl2_scale, reconstruct = morl2_reconstruct(order_Nth, frequency_hz, frequency_sample_rate_hz)
    # Scaled wavelet coefficients
    f_x_cwtm = d1tile_x_d2(reconstruct, cwtm)

    # Reference functions
    sig_inv = np.sum(f_x_cwtm, 0)
    sig_hilbert = signal.hilbert(sig_n)
    sig_theory = 1.*sig_complex
    noise_hilbert = signal.hilbert(noise)

    # # Entropy
    # energy_pdf, entropy_LEE, entropy_SE = energy_pdf_entropy(f_x_cwtm)
    energy_pdf, entropy_LEE, entropy_SE = energy_pdf_entropy(cwtm)

    # Compute SNR from noise directly - still oscillates
    # Creates instability due to variability, specifically at low frequencies
    # Real and imaginary part of noise are the same
    # Fast approach, less vulnerable
    noise_power = np.mean(cwtm_noise.real**2) \
                  *np.ones(len(frequency_hz)) \
                  *(1 + 1j)

    # noise_p2 = np.mean(cwtm_noise.real**2, axis=1) + \
    #            np.mean(cwtm_noise.imag**2, axis=1)*1j
    # noise_p2_mean = np.mean(noise_p2)*np.ones(len(frequency_hz))
    # plt.plot(frequency_hz, noise_p2.real, 'r*',
    #          frequency_hz, noise_p2_mean.real, 'ro',
    #          frequency_hz, noise_power.real, 'k--')
    # plt.plot(frequency_hz, noise_p2.imag, 'b*',
    #          frequency_hz, noise_p2_mean.imag, 'bx',
    #          frequency_hz, noise_power.real, 'ko')
    # plt.show()
    # exit()

    # TODO: Explain 'Flatten' the noise
    cwtm_noise_power_tile = just_tile(noise_power.real, cwtm_noise.shape) + \
                            just_tile(noise_power.imag, cwtm_noise.shape)*1j

    snr = cwtm.real**2/cwtm_noise_power_tile.real + \
          cwtm.imag**2/cwtm_noise_power_tile.imag*1j

    # Build a PDF and entropy of the traditional SNR
    snr_pdf, snr_LEE, snr_SE = snr_pdf_entropy(snr)

    # Modal superposition from largest contributions
    morl2_inv_real2 = np.zeros((len(frequency_hz), len(sig_n)))
    morl2_inv_imag2 = np.zeros((len(frequency_hz), len(sig_n)))

    # super_array = 1.*entropy_SE
    super_array = 1.*snr_SE
    super_array_max = np.max(super_array)
    cutoff = 0*1./2**6
    mode_counter = 0
    print('Max:', super_array_max)
    for j in np.arange(len(frequency_hz)):
        m_cw_real = np.argmax((super_array[j, :].real))
        m_cw_imag = np.argmax((super_array[j, :].imag))
        f_j_hz = frequency_hz[j]
        t_m_off_s = time_s[m_cw_real] + \
                    time_s[m_cw_imag]*1j

        # print(m_cw_real, m_cw_imag)
        # print(t_m_off_s)
        condition1 = super_array[j, m_cw_real] >= cutoff*super_array_max.real
        condition2 = super_array[j, m_cw_imag] >= cutoff*super_array_max.imag
        if condition1 and condition2:
            c_m_amp = cwtm[j, m_cw_real].real + cwtm[j, m_cw_imag].imag*1j
            mode_counter += 1
        else:
            c_m_amp = 0*(1 + 1j)

        morl2_inv_real2[j, :] = \
            inv_morl2_real(order_Nth, time_s, t_m_off_s.real,
                           f_j_hz, c_m_amp.real, frequency_sample_rate_hz)

        # TODO: Remarkable, explain why must use cosine
        morl2_inv_imag2[j, :] = \
            inv_morl2_real(order_Nth, time_s, t_m_off_s.imag,
                           f_j_hz, c_m_amp.imag, frequency_sample_rate_hz)


    print("Number of bands:", len(frequency_hz))
    print("Number of modes in the superposition:", mode_counter)
    sig_superpose = np.sum(morl2_inv_real2, axis=0) + \
                    np.sum(morl2_inv_imag2, axis=0)*1j


    # #### PLOTS
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
    fig_description = 'Wiggle plots, scaled coefficients'
    sig_arr = sig_hilbert.reshape(1, sig_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, f_x_cwtm))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Input", y_color="black")

    fig_number += 1
    fig_description = 'Wiggle plots, unscaled sig coefficients'
    sig_arr = sig_hilbert.reshape(1, sig_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, cwtm))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Input", y_color="C0")

    fig_number += 1
    fig_description = 'Wiggle plots, unscaled noise coefficients'
    sig_arr = noise_hilbert.reshape(1, noise_hilbert.shape[0])
    y_array = np.concatenate((sig_arr, cwtm_noise))
    y_label = np.append(np.array(0), frequency_scaled)
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=y_array, wf_label=y_label, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Noise", y_color="C0")

    fig_number += 1
    fig_description = 'Wiggle plots, scaled entropy'
    xY = 3*order_Nth
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=xY*entropy_SE, wf_label=frequency_scaled, xlim_max=order_Nth,
                               y0_color="C0", y0_label="Shannon Entropy", y_color="C0")

    fig_number += 1
    fig_description = 'Wiggle plots, SNR entropy, noise re cw'
    xY = 3*order_Nth
    plot_wiggles_complex_label(fig_number, xarray=time_scaled,
                               wf_array=xY*snr_SE, wf_label=frequency_scaled, xlim_max=order_Nth,
                               y0_color="C0", y0_label="SNR RbR", y_color="C0")

    plt.show()





