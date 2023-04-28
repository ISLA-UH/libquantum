import numpy as np
import scipy.signal as signal
from libquantum.scales import EPSILON
from libquantum import synthetics, utils
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt

""" Quick plotting routines """


def plot_tdr_sig(sig_wf, sig_time,
                 signal_time_base: str = 'seconds'):
    """
    Waveform
    :param sig_wf:
    :param sig_time:
    :param sig_rms_wf:
    :param sig_rms_time:
    :param signal_time_base:
    :return:
    """

    plt.figure()
    plt.plot(sig_time, sig_wf)
    plt.title('Input waveform')
    plt.xlabel("Time, " + signal_time_base)


def plot_tdr_rms(sig_wf, sig_time,
                 sig_rms_wf, sig_rms_time,
                 signal_time_base: str = 'seconds'):
    """
    Waveform
    :param sig_wf:
    :param sig_time:
    :param sig_rms_wf:
    :param sig_rms_time:
    :param signal_time_base:
    :return:
    """

    plt.figure()
    plt.plot(sig_time, sig_wf)
    plt.plot(sig_rms_time, sig_rms_wf)
    plt.title('Input waveform and RMS')
    plt.xlabel("Time, " + signal_time_base)


def plot_tfr_lin(tfr_power, tfr_frequency, tfr_time,
                 title_str: str = 'TFR, power',
                 signal_time_base: str = 'seconds'):
    """
    TFR in linear power
    :param sig_tfr:
    :param sig_tfr_frequency:
    :param sig_tfr_time:
    :param signal_time_base:
    :return:
    """

    plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_power, cmap='RdBu_r')
    plt.title(title_str)
    plt.ylabel("Frequency, samples per " + signal_time_base)
    plt.xlabel("Time, " + signal_time_base)


def plot_tfr_bits(tfr_power, tfr_frequency, tfr_time,
                  bits_min: float = -8,
                  bits_max: float = 0,
                  title_str: str = 'TFR, top bits',
                  y_scale: str = None,
                  tfr_x_str: str = 'Time, seconds',
                  tfr_y_str: str = 'Frequency, hz',
                  tfr_y_flip: bool = False):
    """
    TFR in bits
    :param tfr_y_flip:
    :param sig_tfr:
    :param sig_tfr_frequency:
    :param sig_tfr_time:
    :param bits_max:
    :param bits_min:
    :param y_scale:
    :param title_str:
    :param tfr_x_str:
    :param tfr_y_str:
    :return: figure
    """

    tfr_bits = 0.5 * np.log2(tfr_power / np.max(tfr_power))

    fig = plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_bits,
                   cmap='RdBu_r',
                   vmin=bits_min, vmax=bits_max, shading='nearest')
    if y_scale is None:
        plt.yscale('linear')
    else:
        plt.yscale('log')

    if tfr_y_flip:
        plt.ylim(np.max(tfr_frequency), np.min(tfr_frequency))
    plt.title(title_str)
    plt.ylabel(tfr_y_str)
    plt.xlabel(tfr_x_str)

    return fig


def plot_st_window_tdr_lin(window, freq_sx, time_fft, signal_time_base: str = 'seconds'):
    plt.figure(figsize=(8, 8))
    for j, freq in enumerate(freq_sx):
        plt.plot(time_fft, np.abs(window[j, :]), label=freq)
    plt.legend()
    plt.title('TDR window, linear')


def plot_st_window_tfr_bits(window, frequency_sx, frequency_fft, signal_time_base: str = 'seconds'):
    plt.figure(figsize=(8, 8))
    for j, freq in enumerate(frequency_sx):
        plt.plot(frequency_fft, np.log2(np.abs(window[j, :]) + EPSILON), label=freq)
    plt.legend()
    plt.title('TFR window, bits')


def plot_st_window_tfr_lin(window, frequency_sx, frequency_fft, signal_time_base: str = 'seconds'):
    plt.figure(figsize=(8, 8))
    for j, freq in enumerate(frequency_sx):
        plt.plot(frequency_fft, np.abs(window[j, :]), label=freq)
    plt.legend()
    plt.title('TFR window, lin')


def signal_gate(wf, t, tmin, tmax, fraction_cosine: float = 0):
    """
    Time gate and apply Tukey window, rectangular is the default

    :param wf: waveform
    :param t: time
    :param tmin: lower time limit
    :param tmax: upper time limit
    :param fraction_cosine: 0 = rectangular, 1 = Hann
    :return:
    """
    index_exclude = np.logical_or(t < tmin, t > tmax)
    index_include = np.logical_and(t >= tmin, t <= tmax)
    wf[index_exclude] = 0.
    wf[index_include] *= signal.windows.tukey(M=index_include.sum(), alpha=fraction_cosine)
    return wf


def oversample_time(time_duration, time_sample_interval, oversample_scale):
    """
    Return an oversampled time by a factor oversample_scale
    :param time_duration:
    :param time_sample_interval:
    :param oversample_scale:
    :return:
    """
    oversample_interval = time_sample_interval / oversample_scale
    number_points = int(time_duration / oversample_interval)
    time_all = np.arange(number_points) * oversample_interval
    return time_all


""" Reference Signatures, starting with the quantized Gabor chirp"""


def quantum_chirp(omega: float,
                  order: float = 12,
                  gamma: float = 0,
                  gauss: bool = True,
                  oversample_scale: int = 2):
    """
    Constructs a tone or a sweep with a gaussian window option and a duration of 2^n points
    :param omega: center frequency < pi. Resets to pi/4 if >= 1 (Nyquist).
    :param order: fractional octave band, sets the Q and duration. Default of 12 for chromatic scale.
    :param gamma: sweep index, could be positive or negative. Default of zero yields the atom.
    :param gauss: Apply the Gauss envelope, True as default. False constructs constant amplitude CW or sweep
    :return:
    """

    if omega >= 0.8*np.pi:
        print("Omega >= 0.8*pi (AA*Nyquist), reset to pi * 2**(-1/N")
        omega = np.pi * 2**(-1/order)

    # Gabor atom specifications
    scale_multiplier = 3 / 4 * np.pi * order

    # Atom scale
    scale = scale_multiplier / omega

    # Chirp index gamma, blueshift
    mu = np.sqrt(1 + gamma ** 2)
    chirp_scale = scale * mu

    # scale multiplier Mc
    window_support_points = 2 * np.pi * chirp_scale
    # scale up
    window_support_pow2 = 2 ** int((np.ceil(np.log2(window_support_points))))

    # Oversample by a factor of two:
    window_support_pow2_oversample = oversample_scale * window_support_pow2

    time0 = np.arange(window_support_pow2_oversample)
    time = time0 - time0[-1] / 2

    chirp_phase = omega * time + 0.5 * gamma * (time / chirp_scale) ** 2
    if gauss:
        chirp_wf_oversample = np.exp(-0.5 * (time / chirp_scale) ** 2 + 1j * chirp_phase)
    else:
        chirp_wf_oversample = np.exp(1j * chirp_phase)

    # Downsample to anti-alias
    chirp_wf = signal.decimate(x=np.real(chirp_wf_oversample), q=oversample_scale) \
               + 1j * signal.decimate(x=np.imag(chirp_wf_oversample), q=oversample_scale)

    return chirp_wf, window_support_pow2


def synth_00(frequency_0: float = 100,
             frequency_1: float = 200,
             frequency_2: float = 400,
             time_start_2: float = 0.25,
             time_stop_2: float = 0.4,
             time_sample_interval: float = 1 / 1000,
             time_duration: float = 1,
             oversample_scale: int = 2):
    """
    Generate three sine waves, oversample and decimate to AA
    Always work with  nondimensionalized units (number of points, Nyquist, etc.)

    :param frequency_0:
    :param frequency_1:
    :param frequency_2:
    :param time_start_2:
    :param time_stop_2:
    :param time_sample_interval:
    :param time_duration:
    :param oversample_scale: oversample synthetic, then decimate by scale
    :param frequency_units:
    :param time_units:
    :return:
    """

    # Oversample, then decimate
    oversample_interval = time_sample_interval / oversample_scale
    number_points = int(time_duration / oversample_interval)
    time_all = np.arange(number_points) * oversample_interval

    # Construct sine waves with unit amplitude [rms * sqrt(2)]
    sin_0 = np.sin(2 * np.pi * frequency_0 * time_all)
    signal_gate(wf=sin_0, t=time_all, tmin=0, tmax=0.5)
    sin_1 = np.sin(2 * np.pi * frequency_1 * time_all)
    signal_gate(wf=sin_1, t=time_all, tmin=0.5, tmax=1)
    sin_2 = np.sin(2 * np.pi * frequency_2 * time_all)
    signal_gate(wf=sin_2, t=time_all, tmin=time_start_2, tmax=time_stop_2)

    # Superpose gated sinusoids
    superpose = sin_0 + sin_1 + sin_2
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_01(a: float = 100,
             b: float = 20,
             f: float = 5,
             time_sample_interval: float = 1 / 1000,
             time_duration: float = 1,
             oversample_scale: int = 2):
    """
    Example Synthetic 1
    :param a:
    :param b:
    :param f:
    :param time_sample_interval:
    :param time_duration:
    :param oversample_scale:
    :return:
    """

    # Oversample, then decimate, essentially an AA filter
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)
    superpose = np.cos(a * np.pi * time_all - b * np.pi * time_all * time_all) + \
                np.cos(4 * np.pi * np.sin(np.pi * f * time_all) + np.pi * 80 * time_all)
    # Taper
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)
    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_02(t1: float = 0.3,
             t2: float = 0.7,
             t3: float = 0.5,
             f1: float = 45,
             f2: float = 75,
             f3: float = 15,
             time_sample_interval: float = 1 / 1000,
             time_duration: float = 1,
             oversample_scale: int = 2):
    """
    Example Synthetic 2
    :param t1:
    :param t2:
    :param t3:
    :param f1:
    :param f2:
    :param f3:
    :param time_sample_interval:
    :param time_duration:
    :param oversample_scale:
    :return:
    """

    t = oversample_time(time_duration, time_sample_interval, oversample_scale)

    pulse1 = np.exp(-35 * np.pi * (t - t1) ** 2) * np.cos(np.pi * f1 * t)
    pulse2 = np.exp(-35 * np.pi * (t - t2) ** 2) * np.cos(np.pi * f1 * t)
    pulse3 = np.exp(-55 * np.pi * (t - t3) ** 2) * np.cos(np.pi * f2 * t)
    pulse4 = np.exp(-45 * np.pi * (t - t3) ** 2) * np.cos(np.pi * f3 * t)

    superpose = pulse1 + pulse2 + pulse3 + pulse4

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


def synth_03(a: float = 30,
             b: float = 40,
             c: float = 150,
             time_sample_interval: float = 1 / 1000,
             time_duration: float = 1,
             oversample_scale: int = 2):
    """

    :param a:
    :param b:
    :param c:
    :param time_sample_interval:
    :param time_duration:
    :param oversample_scale:
    :return:
    """

    # Oversample, then decimate
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)
    superpose = np.cos(20 * np.pi * np.log(a * time_all + 1)) + \
                np.cos(b * np.pi * time_all + c * np.pi * (time_all ** 2))
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)

    # Decimate by same oversample scale, essentially an AA filter
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf)) * time_sample_interval

    return synth_wf, synth_time


""" Fancy test tone with time-domain specifications"""


def well_tempered_tone(frequency_sample_rate_hz: float = 800.,
                       frequency_center_hz: float = 60.,
                       time_duration_s: float = 16,
                       time_fft_s: float = 1.,
                       use_fft_frequency: bool = True,
                       add_noise_taper_aa: bool = False) -> Tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """
    Return a tone of unit amplitude and fixed frequency
    with a constant sample rate
    :param frequency_sample_rate_hz:
    :param frequency_center_hz:
    :param time_duration_s:
    :param time_fft_s: Split the record into segments. Previous example showed 1s duration was adequate
    :param use_fft_frequency:
    :param add_noise_taper_aa:
    :return:
    """

    # The segments determine the spectral resolution
    frequency_resolution_hz = 1 / time_fft_s

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note that int rounds down
    time_duration_nd = 2 ** (int(np.log2(time_duration_s * frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2 ** (int(np.log2(time_fft_s * frequency_sample_rate_hz)))

    # The fft frequencies are set by the duration of the fft
    # In this example we only need the positive frequencies
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1 / frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz - frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz / time_fft_nd

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless center frequency
    frequency_center = frequency_center_hz / frequency_sample_rate_hz
    frequency_center_fft = frequency_center_fft_hz / frequency_sample_rate_hz
    # Dimensionless time (samples)
    time_nd = np.arange(time_duration_nd)
    time_s = time_nd / frequency_sample_rate_hz

    if use_fft_frequency:
        # Construct synthetic tone with 2^n points and max FFT amplitude at exact fft frequency
        mic_sig = np.cos(2 * np.pi * frequency_center_fft * time_nd)
    else:
        # # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
        # # It does NOT return unit amplitude (but it's close)
        mic_sig = np.cos(2 * np.pi * frequency_center * time_nd)

    if add_noise_taper_aa:
        # Add noise
        mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=8.)
        # Taper before AA
        mic_sig *= utils.taper_tukey(mic_sig, fraction_cosine=0.1)
        # Antialias (AA)
        synthetics.antialias_halfNyquist(mic_sig)

    print('WELL TEMPERED TONE SYNTHETIC')
    print('Nyquist frequency:', frequency_sample_rate_hz / 2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of signal points:', time_duration_nd)
    print('log2(points):', np.log2(time_duration_nd))
    print('Number of FFT points:', time_fft_nd)
    print('log2(FFT points):', np.log2(time_fft_nd))

    return mic_sig, time_s, time_fft_nd, frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz


if __name__ == "__main__":
    sig_wf, sig_t = synth_00()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title('Synth 00')

    sig_wf, sig_t = synth_01()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title('Synth 01')

    sig_wf, sig_t = synth_02()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title('Synth 02')

    sig_wf, sig_t = synth_03()
    plt.figure()
    plt.plot(sig_t, sig_wf)
    plt.title('Synth 03')

    plt.show()
