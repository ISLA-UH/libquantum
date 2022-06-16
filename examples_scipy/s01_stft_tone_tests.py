"""
libquantum example: s00_stft_tone_intro.py
Compute STFT TFRs on tones to verify amplitudes
Introduce the concept of a Q-driven STFT

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import synthetics, utils, entropy
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq

print(__doc__)
EVENT_NAME = 'tone test'
alpha = 1

if __name__ == "__main__":
    """
    # The primary goal of standardization is to permit multimodal sensor analysis for different sample rates
    # For a specified signal duration, there is only one key parameter: Order
    
    From scipy code for spectrogram:
    if scaling == 'density': [ psd? ]
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    # PSD multiplies by 2
    # If not psd, it is stft
    if mode == 'stft':
        scale = np.sqrt(scale)
    if mode == 'magnitude':
        Sxx = np.abs(Sxx)
    """

    # In practice, our quest begins either with a signal duration or a frequency of interest
    frequency_sample_rate_hz = 1000.
    frequency_center_hz = 80.
    # Lowest frequency of interest; averaging frequency sets record duration and spectral resolution
    frequency_averaging_hz = 10.
    order_nth = 12.

    # Stage is set
    # Gabor Multiplier
    m_factor = 2*np.sqrt(2*np.log(2))  # M/N
    print("Exact M/N factor:", m_factor)
    print("Approximation to 3pi/pi:", 3*np.pi/4)
    # Use 3pi/4
    number_cycles_per_order = 3*np.pi/4*order_nth

    # Dimensionless time frequency
    frequency_center = frequency_center_hz/frequency_sample_rate_hz
    period_center = 1./frequency_center
    frequency_averaging = frequency_averaging_hz/frequency_sample_rate_hz
    period_averaging = 1./frequency_averaging

    # Set the record duration by the averaging frequency; make a power of 2
    time_duration_nd = 2**(int(np.log2(number_cycles_per_order*period_averaging)))
    # Set the fft duration by the order, make a power of 2
    time_fft_nd = 2**(int(np.log2(number_cycles_per_order*period_center)))
    # Note: int rounds down

    time_nd = np.arange(time_duration_nd)

    # Construct synthetic tone with 2^n points and max unit rms amplitude
    mic_sig = np.sqrt(2)*np.sin(2*np.pi*frequency_center*time_nd)
    # Add noise
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=12.)
    # Taper before AA
    mic_sig *= utils.taper_tukey(mic_sig, fraction_cosine=0.1)
    # Antialias (AA)
    synthetics.antialias_halfNyquist(mic_sig)

    nfft_center = time_fft_nd
    print('\nRequest Order N =', order_nth)
    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Signal frequency, hz:', frequency_center_hz)
    print('Averaging frequency, hz:', frequency_averaging_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')
    print('log2(NDuration):', np.log2(time_duration_nd))
    print('log2(NFFT):', np.log2(nfft_center))

    plt.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    plt.title('Synthetic sinusoid')

    mic_stft_frequency_hz, mic_stft_time_s, mic_stft_psd_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='psd')

    mic_stft_bits = utils.log2epsilon(np.abs(mic_stft_psd_spec))
    # Select plot frequencies
    fmin = frequency_averaging
    fmax = frequency_sample_rate_hz/2  # Nyquist
    pltq.plot_wf_mesh_vert(redvox_id='tone test',
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_nd/frequency_sample_rate_hz,
                           mesh_time=mic_stft_time_s,
                           mesh_frequency=mic_stft_frequency_hz,
                           mesh_panel_b_tfr=mic_stft_bits,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="stft for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()

    exit()


    # Scipy TFR
    # compute raw spectrogram with scipy package
    # Iterate over the various options
    # scaling = 'density', 'spectrum'
    # mode = [‘psd’, ‘complex’, ‘magnitude’, ‘angle’, ‘phase’]


    welch_frequency_hz, Pxx = signal.welch(x=mic_sig,
                                           fs=frequency_sample_rate_hz,
                                           window=('tukey', alpha),
                                           nperseg=nfft_center,
                                           noverlap=nfft_center // 2,
                                           nfft=nfft_center,
                                           detrend='constant',
                                           return_onesided=True,
                                           axis=-1,
                                           scaling='density',
                                           average='mean')

    _, Pxx_spec =             signal.welch(x=mic_sig,
                                           fs=frequency_sample_rate_hz,
                                           window=('tukey', alpha),
                                           nperseg=nfft_center,
                                           noverlap=nfft_center // 2,
                                           nfft=nfft_center,
                                           detrend='constant',
                                           return_onesided=True,
                                           axis=-1,
                                           scaling='spectrum',
                                           average='mean')

    mic_stft_frequency_hz, mic_stft_time_s, mic_stft_psd = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='psd')

    _, _, mic_stft_complex = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='complex')

    _, _, mic_stft_magnitude = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='density',
                           mode='magnitude')

    _, _, mic_stft_psd_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='psd')

    _, _, mic_stft_complex_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='complex')

    _, _, mic_stft_magnitude_spec = \
        signal.spectrogram(x=mic_sig,
                           fs=frequency_sample_rate_hz,
                           window=('tukey', alpha),
                           nperseg=nfft_center,
                           noverlap=nfft_center // 2,
                           nfft=nfft_center,
                           detrend='constant',
                           return_onesided=True,
                           axis=-1,
                           scaling='spectrum',
                           mode='magnitude')

    plt.figure()

    # These appear to be the most predictable
    plt.plot(mic_stft_frequency_hz, np.sqrt(np.average(mic_stft_psd_spec, axis=1)), label='spec, psd')
    plt.plot(welch_frequency_hz, np.sqrt(Pxx_spec), '-.', label='spec, Pxx psd')

    # plt.plot(mic_stft_frequency_hz, np.sqrt(np.average(mic_stft_psd, axis=1)), label='density, psd')
    # plt.plot(welch_frequency_hz, np.sqrt(Pxx), '-.', label='density, Pxx psd')
    # plt.plot(mic_stft_frequency_hz, np.sqrt(np.average(mic_stft_magnitude, axis=1)), label='density, mag')
    # plt.plot(mic_stft_frequency_hz, np.sqrt(np.average(mic_stft_magnitude_spec, axis=1)), label='spec, mag')
    # plt.plot(mic_stft_frequency_hz, utils.mean_columns(np.abs(mic_stft_complex)), label='density, complex')
    plt.plot(mic_stft_frequency_hz, np.sqrt(2)*np.average(np.abs(mic_stft_complex_spec), axis=1), label='spec, complex')

    plt.xlim(frequency_sample_rate_hz-10, frequency_sample_rate_hz+10)
    plt.grid(True)
    plt.legend()

    plt.show()
    # exit()

    # mic_stft_bits = utils.log2epsilon(np.abs(mic_stft))
    # pltq.plot_wf_mesh_vert(redvox_id=station_id_str,
    #                        wf_panel_a_sig=mic_sig,
    #                        wf_panel_a_time=mic_sig_epoch_s,
    #                        mesh_time=mic_stft_time_s,
    #                        mesh_frequency=mic_stft_frequency_hz,
    #                        mesh_panel_b_tfr=mic_stft_bits,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="stft for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)
    #
    # plt.show()
