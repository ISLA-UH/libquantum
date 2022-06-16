"""
libquantum example: s00_stft_tone_intro.py
Compute STFT TFRs on tones to verify amplitudes

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
    # The first step is understanding the foundation: The Fast Fourier Transform
    """

    # In practice, our quest begins with a signal within a record of fixed duration
    # Construct a tone of fixed frequency with a constant sample rate
    frequency_sample_rate_hz = 1000.
    frequency_center_hz = 50.
    time_duration_s = 10.
    # We are going to split the record into segments
    time_fft_s = time_duration_s/16
    # Those segments determine the spectral resolution
    frequency_resolution_hz = 1/time_fft_s

    # We convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless frequency
    frequency_center = frequency_center_hz/frequency_sample_rate_hz

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note: int rounds down
    time_duration_nd = 2**(int(np.log2(time_duration_s*frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2**(int(np.log2(time_fft_s*frequency_sample_rate_hz)))

    # Dimensionless time (samples)
    time_nd = np.arange(time_duration_nd)

    # Construct synthetic tone with 2^n points and max unit rms amplitude embedded in white noise
    mic_sig = np.sqrt(2)*np.sin(2*np.pi*frequency_center*time_nd)
    # Add noise
    mic_sig += synthetics.white_noise_fbits(sig=mic_sig, std_bit_loss=12.)
    # Taper before AA
    mic_sig *= utils.taper_tukey(mic_sig, fraction_cosine=0.1)
    # Antialias (AA)
    synthetics.antialias_halfNyquist(mic_sig)

    # The fft should have sufficient resolution to recover the peak frequency
    nfft_center = time_fft_nd
    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Signal frequency, hz:', frequency_center_hz)
    print('Spectral resolution, hz', frequency_resolution_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')
    print('log2(NDuration):', np.log2(time_duration_nd))
    print('log2(NFFT):', np.log2(nfft_center))

    # Let's compute the FFT of the whole record
    fft_points = len(mic_sig)
    fft_sig_pos = np.fft.rfft(mic_sig)
    # returns correct RMS power level sqrt(2) -> 1
    fft_sig_pos /= fft_points
    fft_frequency_pos = np.fft.rfftfreq(fft_points, d=1/frequency_sample_rate_hz)
    fft_power = 2*np.abs(fft_sig_pos)**2

    # Here is the sine wave and its FFT:
    plt.subplot(211), plt.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    plt.title('Synthetic sinusoid')
    plt.subplot(212), plt.semilogx(fft_frequency_pos, fft_power)

    plt.show()
    exit()

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
    fmin = 1/time_fft_s
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

