import numpy as np
import scipy.signal as signal
from libquantum.scales import EPSILON
from scipy.integrate import cumulative_trapezoid
from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt


def plot_tdr_sig(sig_wf, sig_time,
                 signal_time_base:str='seconds'):
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
                 signal_time_base:str='seconds'):
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
                 title_str: str='TFR, power',
                 signal_time_base: str='seconds'):
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
                  title_str: str='TFR, top bits',
                  y_scale: str = None,
                  tfr_x_str: str = 'Time, seconds',
                  tfr_y_str: str = 'Frequency, hz'):
    """
    TFR in bits
    :param sig_tfr:
    :param sig_tfr_frequency:
    :param sig_tfr_time:
    :param bits_max:
    :param bits_min:
    :param y_scale:
    :param title_str:
    :param tfr_x_str:
    :param tfr_y_str:
    :return:
    """

    tfr_bits = 0.5*np.log2(tfr_power/np.max(tfr_power))

    plt.figure()
    plt.pcolormesh(tfr_time, tfr_frequency, tfr_bits,
                   cmap='RdBu_r',
                   vmin=bits_min, vmax=bits_max)
    if y_scale is None:
        plt.yscale('lin')
    else:
        plt.yscale('log')
    plt.title(title_str)
    plt.ylabel(tfr_y_str)
    plt.xlabel(tfr_x_str)


def plot_st_window_tdr_lin(window, freq_sx, time_fft, signal_time_base: str='seconds'):
    plt.figure(figsize=(8, 8))
    for j, freq in enumerate(freq_sx):
        plt.plot(time_fft, np.abs(window[j, :]), label=freq)
    plt.legend()
    plt.title('TDR window, linear')


def plot_st_window_tfr_bits(window, frequency_sx, frequency_fft, signal_time_base: str='seconds'):
    plt.figure(figsize=(8, 8))
    for j, freq in enumerate(frequency_sx):
        plt.plot(frequency_fft, np.log2(np.abs(window[j, :]) + EPSILON), label=freq)
    plt.legend()
    plt.title('TFR window, bits')


def plot_st_window_tfr_lin(window, frequency_sx, frequency_fft, signal_time_base: str='seconds'):
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
    oversample_interval = time_sample_interval/oversample_scale
    number_points = int(time_duration/oversample_interval)
    time_all = np.arange(number_points)*oversample_interval
    return time_all


def synth_00(frequency_0: float = 100,
             frequency_1: float = 200,
             frequency_2: float = 400,
             time_start_2: float = 0.25,
             time_stop_2: float = 0.4,
             time_sample_interval: float = 1/1000,
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
    oversample_interval = time_sample_interval/oversample_scale
    number_points = int(time_duration/oversample_interval)
    time_all = np.arange(number_points)*oversample_interval

    # Construct sine waves with unit amplitude [rms * sqrt(2)]
    sin_0 = np.sin(2*np.pi*frequency_0*time_all)
    signal_gate(wf=sin_0, t=time_all, tmin=0, tmax=0.5)
    sin_1 = np.sin(2*np.pi*frequency_1*time_all)
    signal_gate(wf=sin_1, t=time_all, tmin=0.5, tmax=1)
    sin_2 = np.sin(2*np.pi*frequency_2*time_all)
    signal_gate(wf=sin_2, t=time_all, tmin=time_start_2, tmax=time_stop_2)

    # Superpose gated sinusoids
    superpose = sin_0 + sin_1 + sin_2
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)

    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf))*time_sample_interval

    return synth_wf, synth_time


def synth_01(a: float = 100,
             b: float = 20,
             f: float = 5,
             time_sample_interval: float = 1/1000,
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

    # Oversample, then decimate
    time_all = oversample_time(time_duration, time_sample_interval, oversample_scale)
    superpose = np.cos(a*np.pi*time_all - b*np.pi*time_all*time_all) + \
                np.cos(4*np.pi * np.sin(np.pi*f*time_all) + np.pi*80*time_all)
    # Taper
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)
    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf))*time_sample_interval

    return synth_wf, synth_time


def synth_02(t1: float = 0.3,
             t2: float = 0.7,
             t3: float = 0.5,
             f1: float = 45,
             f2: float = 75,
             f3: float = 15,
             time_sample_interval: float = 1/1000,
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

    pulse1 = np.exp(-35*np.pi*(t-t1)**2)*np.cos(np.pi*f1*t)
    pulse2 = np.exp(-35*np.pi*(t-t2)**2)*np.cos(np.pi*f1*t)
    pulse3 = np.exp(-55*np.pi*(t-t3)**2)*np.cos(np.pi*f2*t)
    pulse4 = np.exp(-45*np.pi*(t-t3)**2)*np.cos(np.pi*f3*t)

    superpose = pulse1 + pulse2 + pulse3 + pulse4

    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf))*time_sample_interval

    return synth_wf, synth_time


def synth_03(a: float = 30,
             b: float = 40,
             c: float = 150,
             time_sample_interval: float = 1/1000,
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
    superpose = np.cos(20*np.pi*np.log(a*time_all + 1)) + \
                np.cos(b*np.pi*time_all + c*np.pi*(time_all**2))
    signal_gate(wf=superpose, t=time_all, tmin=0, tmax=1, fraction_cosine=0.05)

    # Decimate by same oversample scale
    synth_wf = signal.decimate(x=superpose, q=oversample_scale)
    synth_time = np.arange(len(synth_wf))*time_sample_interval

    return synth_wf, synth_time


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
