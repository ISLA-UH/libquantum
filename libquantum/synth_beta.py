import numpy as np
import scipy.io.wavfile
import scipy.signal as signal
from scipy.signal.windows import tukey, hann
import redvox.common.date_time_utils as dt
import matplotlib.pyplot as plt

"""LAST UPDATED: 20200528 MAG"""


# Auxiliary modules for building stations
def datetime_now_epoch_s() -> float:
    """
    Returns the invocation Unix time in seconds

    :return: The current epoch timestamp as seconds since the epoch UTC
    """
    return dt.datetime_to_epoch_seconds_utc(dt.now())


def datetime_now_epoch_micros() -> float:
    """
    Returns the invocation Unix time in microseconds

    :return: The current epoch timestamp as microseconds since the epoch UTC
    """
    return dt.datetime_to_epoch_microseconds_utc(dt.now())


def norm_max(sig: np.ndarray) -> np.ndarray:
    """
    Normalize signal using absolute max value in signal

    :param sig: array-like with signal waveform
    :return: array-like with normalized signal
    """
    sig_norm = sig/np.max(np.abs(sig))
    return sig_norm


def norm_L1(sig: np.ndarray) -> np.ndarray:
    """
    Normalize signal using L1 Normalization method

    :param sig: array-like with signal waveform
    :return: array-like with normalized signal
    """
    sig /= np.sum(sig)
    return sig


def norm_L2(sig: np.ndarray) -> np.ndarray:
    """
    Normalize signal using L2 Normalization method

    :param sig: array-like with signal waveform
    :return: array-like with normalized signal
    """
    sig /= np.sqrt(np.sum(sig**2))
    return sig


def amplitude_tukey(scaled_time_tau: np.ndarray,
                    fraction_cosine: float = 0.5) -> np.ndarray:
    """
    Return a Tukey window, also known as a tapered cosine window

    :param scaled_time_tau: array-like with time
    :param fraction_cosine: shape parameter of the Tukey window, representing the fraction of the window inside the
        cosine tapered region. Default is 0.5
    :return: np.ndarray with Tukey window
    """
    number_points = np.size(scaled_time_tau)
    amplitude = tukey(number_points, fraction_cosine)
    return amplitude


def amplitude_attenuation(frequency_hz,  # TODO MAG: Complete type
                          alpha_nepers_s2_m,
                          range_m):
    """
    Attenuate amplitude

    :param frequency_hz: TODO MAG: Complete me
    :param alpha_nepers_s2_m:
    :param range_m:
    :return:
    """
    amplitude = np.exp(-alpha_nepers_s2_m*range_m*frequency_hz**2)
    return amplitude


def amplitude_spherical(wavelength_m,
                        range_m):  # TODO MAG: Complete type
    """
    TODO MAG: Complete me

    :param wavelength_m:
    :param range_m:
    :return:
    """
    amplitude = np.ones(range_m.size)
    if range_m > wavelength_m:
        amplitude = 1/range_m
    return amplitude


def amplitude_cylindrical(wavelength_m,  # TODO MAG: Complete type
                          range_m):
    """
    TODO MAG: Complete me

    :param wavelength_m:
    :param range_m:
    :return:
    """
    amplitude = np.ones(range_m.size)
    if range_m > wavelength_m:
        amplitude = 1/np.sqrt(range_m)
    return amplitude


def gt_blast(time_s: np.ndarray,
             time_zero_s: float,
             time_pos_s: float) -> np.ndarray:
    """
    Garces (2019) ground truth blast pulse

    :param time_s: array-like with timestamps
    :param time_zero_s: start time in seconds
    :param time_pos_s: TODO MAG: Complete me
    :return: TODO MAG: Complete me
    """
    # Garces (2019) ground truth blast pulse
    # with the +1, tau is the zero crossing time - time_start renamed to time_zero for first zero crossing.
    # time_start = time_zero - time_pos
    tau = (time_s-time_zero_s)/time_pos_s + 1.
    # Initialize GT
    p_GT = np.zeros(tau.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((0.0 <= tau) & (tau <= 1.))  # ONLY positive pulse
    sigintG17 = np.where((1. < tau) & (tau <= 1 + np.sqrt(6.)))  # GT balanced pulse
    p_GT[sigint1] = (1. - tau[sigint1])
    p_GT[sigintG17] = 1./6. * (1. - tau[sigintG17]) * (1. + np.sqrt(6) - tau[sigintG17]) ** 2.
    return p_GT


def wavelet_hellborne(time_s: np.ndarray,  # TODO MAG: Complete type and return
                      offset_time_s,
                      order_cycles_M,
                      scale_frequency_center_hz):
    """
    Hellborne Gabor wavelet
    From Mallat (2009), the angular lifetime is the product and the scale and sigma.
    Identified as the hellborne lifetime in Garces(2020).

    :param time_s: array-like with timestamps
    :param offset_time_s: offset time in seconds
    :param order_cycles_M:  TODO MAG: Complete me
    :param scale_frequency_center_hz: TODO MAG: Complete me
    :return: TODO MAG: Complete me
    """
    time_shifted_s = time_s - offset_time_s
    # Hellborne Gabor wavelet
    # From Mallat (2009), the angular lifetime is the product and the scale and sigma.
    # Identified as the hellborne lifetime in Garces(2020).
    scale_lifetime_s = order_cycles_M / scale_frequency_center_hz  # Lifetime, sigma
    scale_lifetime_angular_sprad = scale_lifetime_s / (2. * np.pi)  # Lifetime per cycle M/omega
    scale_angular_frequency_radps = 2 * np.pi * scale_frequency_center_hz
    wavelet_amplitude = 1. / (np.sqrt(2*np.pi) * scale_lifetime_angular_sprad)
    wavelet_gauss = wavelet_amplitude * np.exp(-0.5 * (time_shifted_s / scale_lifetime_angular_sprad) ** 2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * scale_angular_frequency_radps * time_shifted_s)
    return wavelet_gabor


# NEW FUNCION FAMILY, 2^n points, centered
def gt_blast_cycle(scale_frequency_center_hz,  # TODO MAG: Complete type and return
                   sample_rate_hz: float):
    """
    Garces (2019) ground truth blast pulse rederived for hellborne testing.
    Centered at the zero crossing time, 2^n points

    :param scale_frequency_center_hz: TODO MAG: Complete me
    :param sample_rate_hz:  sample rate in Hz
    :return: TODO MAG: Complete me
    """
    # Garces (2019) ground truth blast pulse rederived for hellborne testing.
    # Centered at the zero crossing time, 2^n points
    order_cycles_M = np.sqrt(6)/2.  ## <~ srt(2)
    scale_lifetime_s = order_cycles_M / scale_frequency_center_hz
    # 2**n points
    duration_points = int(2**(np.ceil(np.log2(scale_lifetime_s*sample_rate_hz))))
    time_s = np.arange(duration_points)/sample_rate_hz
    time_shifted_s = time_s - time_s[-1]/2.
    scaled_time = time_shifted_s * scale_frequency_center_hz
    tau = 4*scaled_time
    # Initialize GT
    p_GT = np.zeros(scaled_time.size)  # Granstrom-Triangular (GT), 2019
    # Initialize time ranges
    sigint1 = np.where((-1. <= tau) & (tau <= 0.))  # ONLY positive pulse
    sigint2 = np.where((0. < tau) & (tau <= np.sqrt(6.)))  # GT balanced pulse
    p_GT[sigint1] = - tau[sigint1]
    p_GT[sigint2] = -1./6. * tau[sigint2] * (np.sqrt(6) - tau[sigint2]) ** 2.
    return scaled_time, p_GT


def wavelet_hellborne_cycles(order_Nth,  # TODO MAG: Complete type and return
                             order_cycles_M,
                             scale_frequency_center_hz,
                             sample_rate_hz: float):
    """
    Hellborne Gabor wavelet
    From Mallat (2009), the angular lifetime is the product and the scale and sigma.
    Identified as the hellborne lifetime in Garces(2020).

    :param order_Nth: TODO MAG: Complete me
    :param order_cycles_M:
    :param scale_frequency_center_hz:
    :param sample_rate_hz: sample rate in Hz
    :return:
    """

    scale_lifetime_s = order_cycles_M / scale_frequency_center_hz   # Lifetime, sigma
    # 2**n points
    duration_points = int(2**(np.ceil(np.log2(scale_lifetime_s*sample_rate_hz))))
    time_s = np.arange(duration_points)/sample_rate_hz
    time_shifted_s = time_s - time_s[-1]/2.
    scaled_time = time_shifted_s * scale_frequency_center_hz

    scale_lifetime_angular_sprad = scale_lifetime_s / (2. * np.pi)  # Lifetime per cycle, M/omega
    scale_angular_frequency_radps = 2 * np.pi * scale_frequency_center_hz
    bandedge_dB = 10*np.log10(np.exp(1)) / 8. * (order_cycles_M/order_Nth)**2
    wavelet_amplitude = 1. # / (np.sqrt(2*np.pi) * scale_lifetime_angular_sprad)
    wavelet_gauss = wavelet_amplitude * np.exp(-0.5 * (time_shifted_s / scale_lifetime_angular_sprad) ** 2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * scale_angular_frequency_radps * time_shifted_s)
    return scaled_time, wavelet_gauss, wavelet_gabor, bandedge_dB


def wavelet_hellborne_dB(order_Nth,   # TODO MAG: Complete type and return
                         bandedge_dB,
                         scale_frequency_center_hz,
                         sample_rate_hz: float):
    """
    Hellborne Gabor wavelet
    From Mallat (2009), the angular lifetime is the product and the scale and sigma.
    Identified as the hellborne lifetime in Garces(2020).

    :param order_Nth: TODO MAG: Complete me
    :param bandedge_dB:
    :param scale_frequency_center_hz:
    :param sample_rate_hz: sample rate in Hz
    :return:
    """

    order_cycles_M = order_Nth * np.sqrt(4.*bandedge_dB/(5.*np.log10(np.exp(1))))
    scale_lifetime_s = order_cycles_M / scale_frequency_center_hz   # Lifetime, sigma
    # 2**n points
    duration_points = int(2**(np.ceil(np.log2(scale_lifetime_s*sample_rate_hz))))
    time_s = np.arange(duration_points)/sample_rate_hz
    time_shifted_s = time_s - time_s[-1]/2.
    scaled_time = time_shifted_s * scale_frequency_center_hz

    scale_lifetime_angular_sprad = scale_lifetime_s / (2. * np.pi)  # Lifetime per cycle, M/omega
    scale_angular_frequency_radps = 2 * np.pi * scale_frequency_center_hz
    wavelet_amplitude = 1. / (np.sqrt(2*np.pi) * scale_lifetime_angular_sprad)
    wavelet_gauss = wavelet_amplitude * np.exp(-0.5 * (time_shifted_s / scale_lifetime_angular_sprad) ** 2)
    wavelet_gabor = wavelet_gauss * np.exp(1j * scale_angular_frequency_radps * time_shifted_s)
    return scaled_time, wavelet_gauss, wavelet_gabor, order_cycles_M


### BACK TO LEGACY
def chirp_amplitude_hellborne(scaled_time_tau: np.ndarray,  # TODO MAG: Complete type and return
                              symmetry,
                              conv_rate_scaled):
    """
    TODO MAG: Complete me

    :param scaled_time_tau: array-like with time
    :param symmetry:
    :param conv_rate_scaled:
    :return:
    """
    alpha = np.pi  # Attenuation rate, for defining System Q
    if np.abs(symmetry) > 1:
        symmetry = 0.95
        print('Fix symmetry, magnitude must be less than one')
    phase_decay = (2.*scaled_time_tau)**2/(1.-np.abs(symmetry))
    phase_hyper = 1. - symmetry*np.tanh(2.*np.pi*scaled_time_tau*conv_rate_scaled)
    # TODO MAG: fix rounding off errors near |s|=1, conv=1 as system does not know how to do L'Hopitals.
    amplitude = np.exp(-alpha*phase_decay*phase_hyper)
    return amplitude


def chirp_amplitude_hann(scaled_time_tau: np.ndarray) -> np.ndarray:
    """
    Hann window for chirp

    :param scaled_time_tau: array-like with time
    :return: np.ndarray with Hann window
    """
    amplitude = np.cos(np.pi*scaled_time_tau)**2
    return amplitude


def chirp_phase_freq_quad(scaled_phase_psi,   # TODO MAG: Complete type and return
                          scaled_duration_T,
                          scaled_frequency_start,
                          scaled_bandpass_mu,
                          theta_radians):
    """
    TODO MAG: Complete me

    :param scaled_phase_psi:
    :param scaled_duration_T:
    :param scaled_frequency_start:
    :param scaled_bandpass_mu:
    :param theta_radians:
    :return:
    """
    scaled_phase_psi_0 = np.pi*scaled_duration_T
    phase_radians = scaled_frequency_start*(scaled_phase_psi-scaled_phase_psi_0) + \
                  scaled_bandpass_mu*(scaled_phase_psi**2 - scaled_phase_psi_0**2)/(2.*np.pi*scaled_duration_T) + \
                    + theta_radians
    tau_plus_half = scaled_phase_psi/(2.*np.pi*scaled_duration_T)
    scaled_frequency = scaled_frequency_start + 2*scaled_bandpass_mu*tau_plus_half
    return phase_radians, scaled_frequency


def chirp_phase_freq_hann(scaled_phase_psi,  # TODO MAG: Complete type and return
                          scaled_duration_T,
                          hann_harmonic,
                          scaled_frequency_start,
                          scaled_bandpass_mu,
                          theta_radians):
    """
    TODO MAG: Complete me

    :param scaled_phase_psi:
    :param scaled_duration_T:
    :param hann_harmonic:
    :param scaled_frequency_start:
    :param scaled_bandpass_mu:
    :param theta_radians:
    :return:
    """
    scaled_window = scaled_duration_T/hann_harmonic
    scaled_phase_psi_0 = np.pi*scaled_duration_T
    phase_start = (scaled_frequency_start + scaled_bandpass_mu)*(scaled_phase_psi-scaled_phase_psi_0)
    phase_sweep = -scaled_bandpass_mu*scaled_window*\
                  (np.sin(scaled_phase_psi/scaled_window) - np.sin(scaled_phase_psi_0/scaled_window))
    phase_radians = phase_start + phase_sweep + theta_radians
    tau_plus_half = scaled_phase_psi/(2.*np.pi*scaled_duration_T)
    scaled_frequency = scaled_frequency_start + 2*scaled_bandpass_mu*(np.sin(np.pi*hann_harmonic*tau_plus_half)**2)
    return phase_radians, scaled_frequency


def harmonics(phase_radians,  # TODO MAG: Complete type and return
              harmonic_label_int):
    """
    TODO MAG: Complete me

    :param phase_radians:
    :param harmonic_label_int:
    :return:
    """
    # Overtoner. harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = np.arange(harmonic_label_int+1)+1
    for i in range(harmonic_label_int + 1):
        synth += -np.sin((i+1)*phase_radians)
    synth *= 1.  # /(harmonic_label_int+1)
    return synth, mode_number


def sawtooth(phase_radians,
             harmonic_label_int):  # TODO MAG: Complete type and return
    """
    TODO MAG: Complete me

    :param phase_radians:
    :param harmonic_label_int:
    :return:
    """
    # harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = np.arange(harmonic_label_int+1)+1
    for i in range(harmonic_label_int+1):
        synth += ((-1)**(i+1))*np.sin((i+1)*phase_radians)/(i+1)
    synth *= 2./np.pi
    return synth, mode_number


def saw(phase_radians,
        harmonic_label_int):  # TODO MAG: Complete type and return
    """
    TODO MAG: Complete me

    :param phase_radians:
    :param harmonic_label_int:
    :return:
    """
    # harmonic_label_int = 0 is a pure tone.
    # For simpler superposition of image
    synth = np.zeros(phase_radians.size)
    mode_number = np.arange(harmonic_label_int+1)+1
    for i in range(harmonic_label_int+1):
        synth += ((-1)**(i+1))*np.sin((i+1)*phase_radians)/(i+1)
    synth *= 2./np.pi
    return synth

# def saw(phase_radians, harmonic_label_int):
#     # harmonic_label_int = 0 is a pure tone.
#     synth = np.zeros(phase_radians.size)
#     mode_number = np.arange(harmonic_label_int+1)+1
#     for i in range(harmonic_label_int+1):
#         synth += ((-1)**(i+1))*np.sin((i+1)*phase_radians)/(i+1)
#     synth *= 2./np.pi
#     return synth


def square(phase_radians,
           harmonic_label_int):  # TODO MAG: Complete type and return
    """
    TODO MAG: Complete me

    :param phase_radians:
    :param harmonic_label_int:
    :return:
    """
    # harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = 2*np.arange(harmonic_label_int+1) + 1
    for i in range(harmonic_label_int+1):
        synth += -np.sin((2*i + 1)*phase_radians)/(2*i + 1)
    synth *= 4./np.pi
    return synth, mode_number


def harmonics_range(phase_radians, harmonic_label_int, frequency_hz, alpha, range_m):
    # Overtoner. harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = np.arange(harmonic_label_int+1)+1
    for i in range(harmonic_label_int + 1):
        synth += -np.sin((i+1)*phase_radians)*\
                 np.exp(-alpha*range_m*((i+1)*frequency_hz)**2)
    synth *= 1.  #/(harmonic_label_int+1)
    return synth, mode_number

def sawtooth_range(phase_radians, harmonic_label_int, frequency_hz, alpha, range_m):
    # harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = np.arange(harmonic_label_int+1)+1
    for i in range(harmonic_label_int+1):
        synth += ((-1)**(i+1))*np.sin((i+1)*phase_radians)/(i+1)*\
                 np.exp(-alpha*range_m*((i+1)*frequency_hz)**2)
    synth *= 2./np.pi
    return synth, mode_number

def square_range(phase_radians, harmonic_label_int, frequency_hz, alpha, range_m):
    # harmonic_label_int = 0 is a pure tone.
    synth = np.zeros(phase_radians.size)
    mode_number = 2*np.arange(harmonic_label_int+1) + 1
    for i in range(harmonic_label_int+1):
        synth += -np.sin((2*i + 1)*phase_radians)/(2*i + 1)*\
                 np.exp(-alpha*range_m*((2*i + 1)*frequency_hz)**2)
    synth *= 4./np.pi
    return synth, mode_number

def superpose_linear(synth1, synth2):
    synth = synth1 + synth2
    return synth

def superpose_nonlinear(synth1, synth2, power):
    synth = (synth1 + synth2)**power
    return synth

def phase_noise(time_s, mean_phase_noise, std_phase_noise_percent):
    # Add white nose to the synth phase, standard deviation
    # is a percent (%) of 2*pi (full scale). Divided by 100 to get decimal.
    num_samples = time_s.size
    std_phase_noise = 2.*np.pi*std_phase_noise_percent/100.
    phase_noise_radians = np.random.normal(mean_phase_noise, std_phase_noise, size=num_samples)
    return phase_noise_radians

def fft_real(synth, sample_interval_s):
    # FFT for synthetic, by the book
    fft_points = len(synth)
    fft_synth_pos = np.fft.rfft(synth)
    # returns correct RMS power level sqrt(2) -> 1
    fft_synth_pos /= fft_points
    fft_frequency_pos = np.fft.rfftfreq(fft_points, d=sample_interval_s)
    fft_spectral_power_pos_dB = 10.*np.log10(2.*(np.abs(fft_synth_pos))**2.)
    fft_spectral_phase_radians = np.angle(fft_synth_pos)
    return fft_frequency_pos, fft_synth_pos, \
           fft_spectral_power_pos_dB, fft_spectral_phase_radians

# Inverse Fourier Transform from positive FFT of real function, adds zero back
def ifft_real(fft_synth_pos, sample_interval):
    # Assume zero frequency has been removed. Add ot back.
    ifft_synth = np.fft.irfft(fft_synth_pos).real
    fft_points = len(ifft_synth)
    ifft_synth *= fft_points
    return ifft_synth

def fft_complex(synth, sample_interval_s):
    # FFT for synthetic, by the book
    fft_points = len(synth)
    fft_synth = np.fft.fft(synth)
    # returns correct RMS power level
    fft_synth /= fft_points
    fft_frequency = np.fft.fftfreq(fft_points, d=sample_interval_s)
    fft_spectral_bits = np.log2((np.abs(fft_synth)))
    fft_spectral_phase_radians = np.angle(fft_synth)
    return fft_frequency, fft_synth, fft_spectral_bits, fft_spectral_phase_radians

# Inverse Fourier Transform FFT of real function
def ifft_complex(fft_synth_complex):
    ifft_synth = np.fft.ifft(fft_synth_complex)
    fft_points = len(ifft_synth)
    ifft_synth *= fft_points
    return ifft_synth

# Time shifted the FFT before inversion
def fft_time_shift(fft_synth, fft_frequency, time_lead):
    # frequency and the time shift time_lead must have consistent units and be within window
    fft_phase_time_shift = np.exp(-1j*2*np.pi*fft_frequency*time_lead)
    fft_synth *= fft_phase_time_shift
    return fft_synth

# Compute Welch periodogram from spectrogram
def fft_welch_from_Sxx(f_center, Sxx):
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    Welch_Sxx = np.average(Sxx, axis=1)
    Welch_Sxx_db = 10*np.log10(np.abs(Welch_Sxx[1:]))
    # Welch_Sxx_db -= np.max(Welch_Sxx_db) # Normalization
    f_center_nozero = f_center[1:]
    return f_center_nozero, Welch_Sxx_db

def fft_welch_snr_power(f_center, Sxx, Sxx2):
    # Estimate Welch periodogram by adding Sxx and dividing by the number of windows
    Welch_Sxx = np.average(Sxx[1:], axis=1)
    Welch_Sxx /= np.max(Sxx[1:])
    Welch_Sxx2 = np.average(Sxx2[1:], axis=1)
    Welch_Sxx2 /= np.max(Sxx2[1:])
    snr_power = Welch_Sxx2/Welch_Sxx
    snr_frequency = f_center[1:]
    return snr_frequency, snr_power

# Add band-limited noise to a signal at a specified SNR in shannons/bits
def add_noise_01(synth, snr_bits, filter_order, frequency_low_Hz, frequency_high_Hz, sample_rate_Hz):
    synth_max = np.max(np.abs(synth))
    synth_std = np.std(synth)
    # TODO: selection criteria for SNR
    synth_snr_bits = 2.**snr_bits
    std_from_bits = synth_std/np.sqrt(synth_snr_bits)
    synth_noise_white = np.random.normal(0, std_from_bits, size=synth.size)
    # TODO: Error correction for bandedges
    synth_noise_bandpass = bandpass_butter_basic(synth_noise_white, filter_order, frequency_low_Hz, frequency_high_Hz, sample_rate_Hz)
    synth_sig_w_noise = synth + synth_noise_bandpass
    return synth_sig_w_noise

# Tukey 50% (True) and full (Hann) taper window
def taper(synth, tukey_percent_cosine = True):
    taper_window = tukey(synth.size, tukey_percent_cosine/100.)
    synth *= taper_window
    return synth

# Anti-aliasing filter with -3dB at 1/4 of sample rate, 1/2 of Nyquist
def antialias_halfNyquist(synth):
    # Signal frequencies are scaled by Nyquist
    filter_order = 2
    edge_high = 0.5
    b, a = signal.butter(filter_order, edge_high, btype='lowpass')
    synth_anti_aliased = signal.filtfilt(b, a, synth)
    return synth_anti_aliased

# Bandpass filter
def bandpass_butter_2pole(synth, scale_low_Hz, scale_high_Hz, sample_rate_Hz):
    # Signal frequencies are scaled by Nyquist
    filter_order = 2
    nyquist = 0.5*sample_rate_Hz
    edge_low  = scale_low_Hz/nyquist
    edge_high = scale_high_Hz/nyquist
    # TODO: error checking to stay below 1 and above filter instability
    b, a = signal.butter(filter_order, [edge_low, edge_high], btype='bandpass')
    synth_bandpass = signal.filtfilt(b, a, synth)
    return synth_bandpass

def lowpass_butter_2pole(synth, scale_high_Hz, sample_rate_Hz):
    # Signal frequencies are scaled by Nyquist
    filter_order = 2
    nyquist = 0.5*sample_rate_Hz
    edge_high = scale_high_Hz/nyquist
    # TODO: error checking to stay below 1 and above filter instability
    b, a = signal.butter(filter_order, edge_high, btype='lowpass')
    synth_lowpass = signal.filtfilt(b, a, synth)
    return synth_lowpass

def lowpass_butter_Npole(synth, scale_high_Hz, sample_rate_Hz, filter_order_int):
    # Signal frequencies are scaled by Nyquist
    filter_order = filter_order_int
    nyquist = 0.5*sample_rate_Hz
    edge_high = scale_high_Hz/nyquist
    # TODO: error checking to stay below 1 and above filter instability
    b, a = signal.butter(filter_order, edge_high, btype='lowpass')
    synth_lowpass = signal.filtfilt(b, a, synth)
    return synth_lowpass

def bandpass_butter_basic(synth, filter_order, frequency_low_Hz, frequency_high_Hz, sample_rate_Hz):
    # Signal frequencies are scaled by Nyquist
    nyquist = 0.5*sample_rate_Hz
    edge_low  = frequency_low_Hz/nyquist
    edge_high = frequency_high_Hz/nyquist
    # TODO: error checking to stay below 1 and above filter instability
    b, a = signal.butter(filter_order, [edge_low, edge_high], btype='bandpass')
    synth_bandpass = signal.lfilter(b, a, synth)
    return synth_bandpass

def lowpass_cheby1_Npole(synth, scale_high_Hz, sample_rate_Hz, filter_order_int):
    # Signal frequencies are scaled by Nyquist
    filter_order = filter_order_int
    nyquist = 0.5*sample_rate_Hz
    edge_high = scale_high_Hz/nyquist
    # TODO: error checking to stay below 1 and above filter instability
    b, a = signal.cheby1(filter_order, 0.05, edge_high, btype='lowpass')
    synth_lowpass = signal.filtfilt(b, a, synth)
    return synth_lowpass

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

def plot_wf(figure_number, synth_type, title, time, synth, symbol):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    plt.plot(time, synth, symbol)
    plt.title(title, size = text_size)
    plt.grid(True)
    plt.xlabel('Time', size = text_size)
    plt.xlim(np.min(time), np.max(time))
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    plt.tight_layout()
    # fig.savefig(figure_name, dpi = 300)
    return fig

def plot_wf_unix(figure_number, synth_type, title, time, synth, symbol):
    """
    Plots a waveform with unix time in human format

    :param figure_number:
    :param synth_type:
    :param title:
    :param time:
    :param synth:
    :param symbol:
    :return:
    """

    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    plt.plot(time, synth, symbol)
    plt.title(title, size = text_size)
    plt.grid(True)
    plt.xlabel('Time', size = text_size)
    plt.xlim(np.min(time), np.max(time))
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    plt.tight_layout()
    # fig.savefig(figure_name, dpi = 300)
    return fig

def plot_zoom(figure_number, synth_type, title, x_multiplier, scaled_time, synth, symbol):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    # x_multiplier = number of periods
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(211)
    ax1.plot(scaled_time, synth, symbol)
    ax1.set_title(title, size = text_size)
    ax1.grid(True)
    ax1.set_xlim(-x_multiplier, x_multiplier)
    # ax1.autoscale(enable=True, axis='x', tight=True)
    # ax1.autoscale(enable=True, axis='y', tight=True)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)
    ax2 = plt.subplot(212)
    ax2.plot(scaled_time, synth, symbol)
    ax2.grid(True)
    ax2.xlim(-0.5, 0.5)
    # ax1.autoscale(enable=True, axis='y', tight=True)
    ax2.set_xlabel(r'$t/MT_{c}$', size = text_size)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)
    # fig.savefig(figure_name, dpi = 300)
    return fig

def plot_wf_fft(figure_number, synth_type, title, scaled_time, synth,
                fft_frequency_scaled, fft_abs_amplitude,
                xscale_fft, xlim_fft, yscale_fft, ylim_fft, ylabel_fft = 'dB'):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    text_size = 10
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(211)
    ax1.plot(scaled_time, synth)
    ax1.set_title(title, size = text_size)
    ax1.set_xlabel('Time (s)')
    #ax1.text(0, np.min(synth), r'$t/T_{c}$', size = text_size, horizontalalignment='center', verticalalignment='bottom')
    ax1.grid(True)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.tick_params(axis='both', which='major', labelsize=text_size)
    ax1.tick_params(axis='both', which='minor', labelsize=text_size)
    ax2 = plt.subplot(212)
    # ax2.plot(fft_frequency_scaled, fft_abs_amplitude, 'o', color='#ff7f0e')
    # ax2.plot(fft_frequency_scaled, fft_abs_amplitude, '-.', color='#ff7f0e')
    ax2.plot(fft_frequency_scaled, fft_abs_amplitude, '-', color='#ff7f0e')
    ax2.grid(True, which='both')
    ax2.set_xlabel('Hz', size = text_size)
    # ax2.set_xlabel(r'$f/f_{c}$', size = text_size)
    ax2.set_ylabel(ylabel_fft, size = text_size)
    ax2.set_xscale(xscale_fft)
    ax2.set_yscale(yscale_fft)
    ax2.set_xlim(xlim_fft)
    ax2.set_ylim(ylim_fft)
    # ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.tick_params(axis='both', which='major', labelsize=text_size)
    ax2.tick_params(axis='both', which='minor', labelsize=text_size)
    # fig.savefig(figure_name, dpi = 300)
    return fig

def plot_wf_fft_complex(figure_number, synth_type, title, scaled_time, synth_complex,
                fft_frequency_scaled, fft_synth_complex,
                xscale_fft, xlim_fft, yscale_fft, ylim_fft, ylabel_fft = 'dB'):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    text_size = 10
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(211)
    ax1.plot(scaled_time, synth_complex.real, scaled_time, synth_complex.imag)
    ax1.set_title(title, size = text_size)
    ax1.set_xlabel('Time (s)')
    #ax1.text(0, np.min(synth), r'$t/T_{c}$', size = text_size, horizontalalignment='center', verticalalignment='bottom')
    ax1.grid(True)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.tick_params(axis='both', which='major', labelsize=text_size)
    ax1.tick_params(axis='both', which='minor', labelsize=text_size)
    ax2 = plt.subplot(212)
    ax2.plot(fft_frequency_scaled, fft_synth_complex, '-', color='#ff7f0e')
    # ax2.plot(fft_frequency_scaled, 20*np.log10(np.abs(fft_synth_complex.imag)), '-.', color='#ff7f0e')
    # ax2.plot(fft_frequency_scaled, fft_abs_amplitude, '-', color='#ff7f0e')
    ax2.grid(True, which='both')
    ax2.set_xlabel('Hz', size = text_size)
    # ax2.set_xlabel(r'$f/f_{c}$', size = text_size)
    ax2.set_ylabel(ylabel_fft, size = text_size)
    ax2.set_xscale(xscale_fft)
    ax2.set_yscale(yscale_fft)
    ax2.set_xlim(xlim_fft)
    ax2.set_ylim(ylim_fft)
    # ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.tick_params(axis='both', which='major', labelsize=text_size)
    ax2.tick_params(axis='both', which='minor', labelsize=text_size)
    # fig.savefig(figure_name, dpi = 300)
    return fig

def plot_wf_spect(figure_number, synth_type, title, time_s, synth, nfft,
                  sample_rate_Hz, f_center, t_center, Sxx_abs, yscale_f,
                  tmin, tmax, fmin, fmax, cmin, cmax):

    # figure number = integer
    # synth_type = string, for saving
    # title = string, for figure
    # time_s = time in seconds, np.array
    # synth = waveform np.array
    # nfft = 2^n int, 256 nominal
    # sample_rate_HZ = float
    # f_center = frequency from scipy.spectrogram, np.array
    # t_center = time from scipy.spectrogram, np. array
    # Sxx_abs = np.array, spectral power in dB = 10*np.log10(np.abs(Sxx))
    # yscale_f = 'log' or 'linear'
    # tmin = min time display
    # tmax = max time display
    # fmin = min frequency display
    # fmax = max frequency display
    # cmin = cmax - db_range (db for colorbar label) (int)
    # cmax = np.max(Sxx_abs) (db for colorbar label) (int)

    figure_size_x, figure_size_y, text_size = plot_parameters()
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(212)
    ax1.plot(time_s, synth)
    ax1.set_xlim(tmin, tmax)
    ax1.set_xlabel('Time, s', size = text_size)
    ax1.grid(True)
    # ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)
    # ax1.tick_params(axis='both', which='minor', labelsize=text_size)
    ax2 = plt.subplot(211)
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # Assume zero frequency is removed to prevent log(0) exception.
    f_half_bin = f_center[1]/2.
    f_edge1 = f_center[1:]-f_half_bin
    f_edge = np.append(f_edge1, f_center[-1]+f_half_bin)
    t_half_bin_scaled = 1./sample_rate_Hz*nfft/2.
    t_edge1 = t_center - t_half_bin_scaled
    t_edge = np.append(t_edge1, t_center[-1] + t_half_bin_scaled)
    # Construct color map
    sc = ax2.pcolormesh(t_edge, f_edge, Sxx_abs, vmin = cmin, vmax = cmax, cmap='inferno')
    ax2.set_title(title, size = text_size)
    # fig.colorbar(sc, orientation = 'horizontal')
    ax2.get_xaxis().set_ticklabels([])
    ax2.set_ylabel('Frequency, Hz', size=text_size)
    ax2.set_yscale(yscale_f)
    ax2.set_xlim(tmin, tmax)
    ax2.set_ylim(fmin, fmax)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)
    return fig

def plot_wf_spect_frequency_scaled(figure_number, synth_type, title, scaled_time, synth, nfft,
                    sample_rate_Hz, frequency_hz_center, f_center, t_center, Sxx_abs, yscale_f,
                    tmin, tmax, fmin, fmax, cmin, cmax):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    ax1 = plt.subplot(212)
    ax1.plot(scaled_time, synth)
    ax1.set_xlim(tmin, tmax)
    ax1.set_xlabel(r'$t/T_{c}$', size = text_size)
    ax1.grid(True)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.tick_params(axis='both', which='both', labelsize=text_size)
    ax2 = plt.subplot(211)
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # Assume zero frequency is removed to prevent log(0) exception.
    f_half_bin = f_center[1]/2.
    f_edge1 = f_center[1:]-f_half_bin
    f_edge = np.append(f_edge1, f_center[-1]+f_half_bin)
    f_edge /= frequency_hz_center
    t_center_scaled = t_center*frequency_hz_center - np.max(scaled_time)
    t_half_bin_scaled = frequency_hz_center/sample_rate_Hz*nfft/2.
    t_edge1 = t_center_scaled - t_half_bin_scaled
    t_edge = np.append(t_edge1, t_center_scaled[-1] + t_half_bin_scaled)
    # Construct color map
    sc = ax2.pcolormesh(t_edge, f_edge, Sxx_abs, vmin = cmin, vmax = cmax, cmap='inferno')
    ax2.set_title(title, size = text_size)
    # fig.colorbar(sc, orientation = 'horizontal')
    ax2.get_xaxis().set_ticklabels([])
    ax2.set_ylabel(r'$f/f_{c}$', size = text_size)
    ax2.set_yscale(yscale_f)
    ax2.set_xlim(tmin, tmax)
    ax2.set_ylim(fmin, fmax)
    ax2.tick_params(axis='both', which='both', labelsize=text_size)
    return fig


def plot_spect(figure_number, synth_type, title, nfft,
                             sample_rate_Hz, f_center, t_center, Sxx_abs, yscale_f,
                             tmin, tmax, fmin, fmax, cmin, cmax):
    # figure_number, synth_type, title, scaled_time, wf, nfft,
    #                   sample_rate_Hz, frequency_hz_center, f_center, t_center, Sxx_abs, yscale_f,
    #                   tmin, tmax, fmin, fmax, cmin, cmax, color_label = 'dB'
    figure_size_x, figure_size_y, text_size = plot_parameters()
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # Assume zero frequency is removed to prevent log(0) exception.
    f_half_bin = f_center[1]/2.
    f_edge1 = f_center[1:]-f_half_bin
    f_edge = np.append(f_edge1, f_center[-1]+f_half_bin)
    t_half_bin_scaled = 1./sample_rate_Hz*nfft/2.
    t_edge1 = t_center - t_half_bin_scaled
    t_edge = np.append(t_edge1, t_center[-1] + t_half_bin_scaled)
    # Construct color map
    cbar = plt.pcolormesh(t_edge, f_edge, Sxx_abs, vmin = cmin, vmax = cmax, cmap='inferno')
    ax = fig.colorbar(cbar, fraction=0.04, pad=0.01)
    ax.set_label('dB', size=text_size-4)
    plt.title(title, size=text_size)
    plt.ylabel('Frequency, Hz', size=text_size)
    plt.xlabel('Time, s', size=text_size)
    plt.yscale(yscale_f)
    plt.xlim(tmin, tmax)
    plt.ylim(fmin, fmax)
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    return fig

def plot_spect_angle(figure_number, synth_type, title, nfft,
               sample_rate_Hz, f_center, t_center, Sxx_abs, yscale_f,
               tmin, tmax, fmin, fmax, cmin, cmax):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # Assume zero frequency is removed to prevent log(0) exception.
    f_half_bin = f_center[1]/2.
    f_edge1 = f_center[1:]-f_half_bin
    f_edge = np.append(f_edge1, f_center[-1]+f_half_bin)
    t_half_bin_scaled = 1./sample_rate_Hz*nfft/2.
    t_edge1 = t_center - t_half_bin_scaled
    t_edge = np.append(t_edge1, t_center[-1] + t_half_bin_scaled)
    # Construct color map
    cbar = plt.pcolormesh(t_edge, f_edge, Sxx_abs, vmin = cmin, vmax = cmax, cmap='inferno')
    ax = fig.colorbar(cbar, fraction=0.04, pad=0.01)
    ax.set_label('Radians', size=text_size-4)
    plt.title(title, size=text_size)
    plt.ylabel('Frequency, Hz', size=text_size)
    plt.xlabel('Time, s', size=text_size)
    plt.yscale(yscale_f)
    plt.xlim(tmin, tmax)
    plt.ylim(fmin, fmax)
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    return fig

def plot_spect_frequency_scaled(figure_number, synth_type, title, scaled_time, wf, nfft,
                  sample_rate_Hz, frequency_hz_center, f_center, t_center, Sxx_abs, yscale_f,
                  tmin, tmax, fmin, fmax, cmin, cmax, color_label = 'dB'):
    figure_size_x, figure_size_y, text_size = plot_parameters()
    figure_name = './figures/'+ synth_type +'.png'
    fig = plt.figure(figure_number, figsize=(figure_size_x, figure_size_y))
    # Time is in the center of the window, frequency is in the fft coefficient center.
    # pcolormesh must provide corner coordinates, so there will be an offset from step noverlap step size.
    # Assume zero frequency is removed to prevent log(0) exception.
    f_half_bin = f_center[1]/2.
    f_edge1 = f_center[1:]-f_half_bin
    f_edge = np.append(f_edge1, f_center[-1]+f_half_bin)
    f_edge /= frequency_hz_center
    t_center_scaled = t_center*frequency_hz_center - np.max(scaled_time)
    t_half_bin_scaled = frequency_hz_center/sample_rate_Hz*nfft/2.
    t_edge1 = t_center_scaled - t_half_bin_scaled
    t_edge = np.append(t_edge1, t_center_scaled[-1] + t_half_bin_scaled)
    # Construct color map
    cbar = plt.pcolormesh(t_edge, f_edge, Sxx_abs, vmin = cmin, vmax = cmax, cmap='inferno')
    # ax = fig.colorbar(cbar, fraction=0.046, pad=0.04)
    ax = fig.colorbar(cbar, fraction=0.04, pad=0.01)
    ax.set_label(color_label, size = text_size-4)
    plt.title(title, size = text_size)
    plt.xlabel(r'$t/T_{c}$', size = text_size)
    plt.ylabel(r'$f/f_{c}$', size = text_size)
    plt.yscale(yscale_f)
    plt.xlim(tmin, tmax)
    plt.ylim(fmin, fmax)
    plt.tick_params(axis='both', which='both', labelsize=text_size)
    # plt.tick_params(axis='both', which='minor', labelsize=text_size)
    #fig.savefig(figure_name, dpi = 300)
    return fig

def save_to_8k_wav(synth, synth_filename):
    # Save to 8 kHz .wav file
    # Export to wav in scratch directory
    synth_dir = '../../scrap/'
    wav_sample_rate_Hz = 8000
    export_filename = synth_filename + '_8k.wav'
    synth_wav = 0.9*np.real(synth)/np.max(np.abs((np.real(synth))))
    scipy.io.wavfile.write(export_filename, wav_sample_rate_Hz, synth_wav)

def save_to_16k_wav(synth, synth_filename):
    # Save to 8 kHz .wav file
    # Export to wav in scratch directory
    synth_dir = '../../scrap/'
    wav_sample_rate_Hz = 16000
    export_filename = synth_filename + '_16k.wav'
    synth_wav = 0.9*np.real(synth)/np.max(np.abs((np.real(synth))))
    scipy.io.wavfile.write(export_filename, wav_sample_rate_Hz, synth_wav)

def save_to_48k_wav(synth, synth_filename):
    # Save to 44.1 kHz .wav file
    # Export to wav in scratch directory
    synth_dir = '../../scrap/'
    wav_sample_rate_Hz = 48000
    export_filename = synth_filename + '_48k.wav'
    synth_wav = 0.9*np.real(synth)/np.max(np.abs((np.real(synth))))
    scipy.io.wavfile.write(export_filename, wav_sample_rate_Hz, synth_wav)

def save_to_96k_wav(synth, synth_filename):
    # Save to 44.1 kHz .wav file
    # Export to wav in scratch directory
    synth_dir = '../../scrap/'
    wav_sample_rate_Hz = 96000
    export_filename = synth_filename + '_96k.wav'
    synth_wav = 0.9*np.real(synth)/np.max(np.abs((np.real(synth))))
    scipy.io.wavfile.write(export_filename, wav_sample_rate_Hz, synth_wav)

def save_to_192k_wav(synth, synth_filename):
    # Save to 44.1 kHz .wav file
    # Export to wav in scratch directory
    # synth_dir = '../../scrap/'
    wav_sample_rate_Hz = 192000
    export_filename = synth_filename + '_192k.wav'
    synth_wav = 0.9*np.real(synth)/np.max(np.abs((np.real(synth))))
    scipy.io.wavfile.write(export_filename, wav_sample_rate_Hz, synth_wav)