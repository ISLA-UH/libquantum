import numpy as np
import matplotlib.pyplot as plt
from libquantum import scales
from libquantum import atoms


def print_scales_to_screen(scale_order_input: float, scale_base_input: float,
                           scale_ref_input: float, scale_sample_interval_input: float, scale_high_input: float):
    scale_order, scale_base, scale_band_number, \
    scale_ref, scale_center_algebraic, scale_center_geometric, \
    scale_start, scale_end = \
        scales.band_periods_nyquist(scale_order_input, scale_base_input,
                                    scale_ref_input,
                                    scale_sample_interval_input, scale_high_input)

    print('Scale Order:', scale_order)
    print('Reference scale:', scale_ref)
    print('%8s%12s%12s%12s%12s' % ('Band #', 'Center Geo', 'Center Alg', 'Lower', 'Upper'))
    for i in range(scale_band_number.size):
        print('%+8i%12.4e%12.4e%12.4e%12.4e' % (int(scale_band_number[i]),
                                                scale_center_geometric[i], scale_center_algebraic[i],
                                                scale_start[i], scale_end[i]))


def print_frequencies_to_screen(frequency_order_input: float, frequency_base_input: float,
                                frequency_ref_input: float,
                                frequency_low_input: float, frequency_sample_rate_input: float):
    frequency_order, frequency_base, frequency_band_number, \
    frequency_ref, frequency_center_algebraic, frequency_center_geometric, \
    frequency_start, frequency_end = \
        scales.band_frequencies_nyquist(frequency_order_input, frequency_base_input,
                                        frequency_ref_input,
                                        frequency_low_input, frequency_sample_rate_input)

    print('Scale Order:', frequency_order)
    print('Reference frequency:', frequency_ref)
    print('%8s%12s%12s%12s%12s' % ('Band #', 'Center Geo', 'Center Alg', 'Lower', 'Upper'))
    for i in range(frequency_band_number.size):
        print('%+8i%12.4e%12.4e%12.4e%12.4e' % (int(frequency_band_number[i]),
                                                frequency_center_geometric[i], frequency_center_algebraic[i],
                                                frequency_start[i], frequency_end[i]))


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


def plot_wf(figure_number, synth_type, title, time, synth, symbol):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    plt.plot(time, synth, symbol)
    plt.title(title, size=text_size)
    plt.grid(True)
    plt.xlabel('Time, s', size=text_size)
    plt.xlim(np.min(time), np.max(time))
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    plt.tight_layout()
    # fig.savefig(figure_name, dpi = 300)
    return fig


def plot_spect_complex_easy(title_str: str, time: np.ndarray, signal: np.ndarray, frequency: np.ndarray, cwt_complex: np.ndarray):
    plt.figure()
    plt.subplot(3, 1, 1), plt.pcolormesh(time, frequency, cwt_complex.real, cmap='viridis', shading='auto')
    plt.yscale('log')
    plt.ylabel('real, Hz')
    plt.title(title_str)
    plt.subplot(3, 1, 2), plt.pcolormesh(time, frequency, cwt_complex.imag, cmap='viridis', shading='auto')
    plt.yscale('log')
    plt.ylabel('imag, Hz')
    plt.subplot(3, 1, 3), plt.plot(time, signal)
    plt.xlim(time[0], time[-1])
    plt.xlabel('Time, s')
    plt.ylabel('norm')
    plt.grid()


def plot_spect_abs_easy(title_str: str, time: np.ndarray, signal: np.ndarray, time_spect: np.ndarray, frequency: np.ndarray, cqt_abs: np.ndarray):
    plt.figure()
    plt.subplot(2, 1, 1), plt.pcolormesh(time_spect, frequency, cqt_abs, cmap='viridis', shading='auto')
    plt.yscale('log')
    plt.ylabel('Hz')
    plt.title(title_str)
    plt.subplot(2, 1, 2), plt.plot(time, signal)
    plt.xlim(time_spect[0], time_spect[-1])
    plt.xlabel('Time, s')
    plt.ylabel('norm')
    plt.grid()


def plot_redshift(cwt, cwt_red, cwt_blu, time_s, frequency_cwt_hz):
    snr_mean, snr_mean_LEE, snr_mean_SE = atoms.snr_entropy(cwt)
    snr_mean_r, snr_mean_LEE_r, snr_mean_SE_r = atoms.snr_entropy(cwt_red)
    snr_mean_b, snr_mean_LEE_b, snr_mean_SE_b = atoms.snr_entropy(cwt_blu)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.pcolormesh(time_s, frequency_cwt_hz, snr_mean_LEE, cmap='inferno', shading='auto')
    plt.yscale('log')
    plt.title('SNR LEE')
    plt.clim(0, np.max(snr_mean_LEE))
    plt.colorbar(orientation="horizontal")
    plt.subplot(1, 2, 2)
    plt.pcolormesh(time_s, frequency_cwt_hz, snr_mean_LEE_r - snr_mean_LEE_b, cmap='PuOr', shading='auto')
    plt.yscale('log')
    plt.title('SNR LEE R-B')
    plt.clim(-2, 2)
    plt.colorbar(orientation="horizontal")
