"""
libquantum example: s01_tone_fft_welch.py
Compute Welch power spectral density (PSD) on simple tone to verify amplitudes
Case study:
Sinusoid input with unit amplitude
Validate:
Welch power averaged over the signal duration is 1/2
RMS amplitude = 1/sqrt(2)

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from libquantum import benchmark_signals

print(__doc__)

# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 0

if __name__ == "__main__":
    """
    Average the Fast Fourier Transform (FFT) over sliding windows
    using the Welch method
    """

    # Construct a tone of fixed frequency with a constant sample rate
    # For this example, no noise, taper, or AA
    # In the first example (FFT), the nominal signal duration was 1s.
    # In this example the nominal signal duration is 16s, with averaging (fft) window duration of 1s.
    frequency_tone_hz = 60
    [mic_sig, time_s, time_fft_nd,
     frequency_sample_rate_hz, frequency_center_fft_hz, frequency_resolution_fft_hz] = \
        benchmark_signals.well_tempered_tone(frequency_center_hz=frequency_tone_hz,
                                             frequency_sample_rate_hz=800,
                                             time_duration_s=16,
                                             time_fft_s=1,
                                             use_fft_frequency=True,
                                             add_noise_taper_aa=False)

    # Computed and nominal values
    mic_sig_rms = np.std(mic_sig)
    mic_sig_rms_nominal = 1/np.sqrt(2)

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, power_welch_spectrum = signal.welch(x=mic_sig,
                                                            fs=frequency_sample_rate_hz,
                                                            window=('tukey', alpha),
                                                            nperseg=time_fft_nd,
                                                            noverlap=time_fft_nd // 2,
                                                            nfft=time_fft_nd,
                                                            detrend='constant',
                                                            return_onesided=True,
                                                            axis=-1,
                                                            scaling='spectrum',
                                                            average='mean')

    # The density option is the spectrum divided by the spectral resolution.
    # The density option also depends more on the window type, can verify by varying alpha.
    _, power_welch_density = signal.welch(x=mic_sig,
                                          fs=frequency_sample_rate_hz,
                                          window=('tukey', alpha),
                                          nperseg=time_fft_nd,
                                          noverlap=time_fft_nd // 2,
                                          nfft=time_fft_nd,
                                          detrend='constant',
                                          return_onesided=True,
                                          axis=-1,
                                          scaling='density',
                                          average='mean')

    print('Welch returns only the positive frequencies')
    print('len(Pxx):', len(frequency_welch_hz))

    # The spectrum option returns the rms, which for a tone will be 1/sqrt(2)
    fft_rms_welch = np.sqrt(power_welch_spectrum) / mic_sig_rms
    fft_rms_welch_psd = np.sqrt(frequency_resolution_fft_hz*power_welch_density) / mic_sig_rms

    print('\n*** SUMMARY: Welch spectral power estimates for a constant-frequency tone  ***')
    print('The Welch spectral estimate averages the FFT over overlapping windows.')
    print('For the Welch spectrum scaling, the RMS amplitude of a tone is sqrt(P**2) = 1/sqrt(2) = STD(signal)')
    print('The Welch density scaling is divided by the spectral resolution')

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(9, 4))
    ax1.plot(time_s, mic_sig)
    ax1.set_title('Synthetic CW, with taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_welch_hz, fft_rms_welch, label='Welch spectrum')
    ax2.semilogx(frequency_welch_hz, fft_rms_welch_psd, '.-', label='Welch density')
    ax2.set_title('Welch PSD, f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('SQRT(power ave)/STD(signal)')
    ax2.grid(True)
    ax2.legend()

    plt.show()

