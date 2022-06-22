"""
libquantum example: s01_tone_welch_stft.py
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

print(__doc__)

# alpha: Shape parameter of the Tukey window, representing the fraction of the window inside the cosine tapered region.
# If zero, the Tukey window is equivalent to a rectangular window.
# If one, the Tukey window is equivalent to a Hann window.
alpha = 1

if __name__ == "__main__":
    """
    Average the Fast Fourier Transform (FFT) over sliding windows
    using the Welch method
    """

    # Construct a tone of fixed frequency with a constant sample rate
    frequency_sample_rate_hz = 800.
    frequency_center_hz = 60.
    time_duration_s = 16
    # Split the record into segments. Previous example showed 1s duration was adequate
    time_fft_s = time_duration_s/16  # 1 second nominal
    # The segments determine the spectral resolution
    frequency_resolution_hz = 1/time_fft_s

    # The FFT efficiency is based on powers of 2; it is always possible to pad with zeros.
    # Set the record duration, make a power of 2. Note that int rounds down
    time_duration_nd = 2**(int(np.log2(time_duration_s*frequency_sample_rate_hz)))
    # Set the fft duration, make a power of 2
    time_fft_nd = 2**(int(np.log2(time_fft_s*frequency_sample_rate_hz)))

    # The fft frequencies are set by the duration of the fft
    # In this example we only need the positive frequencies
    frequency_fft_pos_hz = np.fft.rfftfreq(time_fft_nd, d=1/frequency_sample_rate_hz)
    fft_index = np.argmin(np.abs(frequency_fft_pos_hz-frequency_center_hz))
    frequency_center_fft_hz = frequency_fft_pos_hz[fft_index]
    frequency_resolution_fft_hz = frequency_sample_rate_hz/time_fft_nd

    # Convert to dimensionless time and frequency, which is typically used in mathematical formulas.
    # Scale by the sample rate.
    # Dimensionless center frequency
    frequency_center = frequency_center_hz/frequency_sample_rate_hz
    frequency_center_fft = frequency_center_fft_hz/frequency_sample_rate_hz
    # Dimensionless time (samples)
    time_nd = np.arange(time_duration_nd)

    # Construct synthetic tone with 2^n points and max FFT amplitude at exact fft frequency
    mic_sig = np.cos(2*np.pi*frequency_center_fft*time_nd)
    # # Compare to synthetic tone with 2^n points and max FFT amplitude NOT at exact fft frequency
    # # It does NOT return unit amplitude
    # mic_sig = np.sin(2*np.pi*frequency_center*time_nd)

    print('Nyquist frequency:', frequency_sample_rate_hz/2)
    print('Nominal signal frequency, hz:', frequency_center_hz)
    print('FFT signal frequency, hz:', frequency_center_fft_hz)
    print('Nominal spectral resolution, hz', frequency_resolution_hz)
    print('FFT spectral resolution, hz', frequency_resolution_fft_hz)
    print('Number of signal points:', time_duration_nd)
    print('log2(points):', np.log2(time_duration_nd))
    print('Number of FFT points:', time_fft_nd)
    print('log2(FFT points):', np.log2(time_fft_nd))

    # Compute the Welch PSD; averaged spectrum over sliding windows
    frequency_welch_hz, Pxx_spec = signal.welch(x=mic_sig,
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

    # # The density option does not return the rms amplitude for a CW unless it is multiplied by the spectral resolution
    # frequency_welch_hz, Pxx_spec = signal.welch(x=mic_sig,
    #                                             fs=frequency_sample_rate_hz,
    #                                             window=('tukey', alpha),
    #                                             nperseg=time_fft_nd,
    #                                             noverlap=time_fft_nd // 2,
    #                                             nfft=time_fft_nd,
    #                                             detrend='constant',
    #                                             return_onesided=True,
    #                                             axis=-1,
    #                                             scaling='density',
    #                                             average='mean')

    print('Welch returns only the positive frequencies')
    print('len(Pxx):', len(frequency_welch_hz))

    # The spectrum option returns the rms, which for a tone will be scaled by sqrt(2)
    fft_rms_abs = np.sqrt(2)*np.sqrt(np.abs(Pxx_spec))
    print('fft_rms_abs[fc]:', fft_rms_abs[fft_index])

    # Show the waveform and the averaged FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 5))
    ax1.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    ax1.set_title('Synthetic CW, no taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_fft_pos_hz, fft_rms_abs)
    ax2.set_title('Welch FFT (RMS), f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('FFT RMS * sqrt(2)')
    ax2.grid(True)

    plt.show()

