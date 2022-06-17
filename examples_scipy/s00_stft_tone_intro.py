"""
libquantum example: s00_stft_tone_intro.py
Compute FFT on simple tones to verify amplitudes

"""
import numpy as np
import matplotlib.pyplot as plt

print(__doc__)
EVENT_NAME = 'tone test'

if __name__ == "__main__":
    """
    # The first step is understanding the foundation: The Fast Fourier Transform
    """

    # In practice, our quest begins with a signal within a record of fixed duration.
    # Construct a tone of fixed frequency with a constant sample rate
    frequency_sample_rate_hz = 800.
    frequency_center_hz = 60.
    time_duration_s = 1
    # Use the whole record
    time_fft_s = time_duration_s
    # It determines the spectral resolution
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
    print('Number of FFT points:', time_duration_nd)
    print('log2(FFT points):', np.log2(time_duration_nd))

    # Compute the RFFT of the whole record
    fft_sig_pos = np.fft.rfft(mic_sig)
    # Compare to the FFT
    fft_sig = np.fft.fft(mic_sig)
    print('RFFT returns only the positive frequencies')
    print('len(FFT):', len(fft_sig))
    print('len(RFFT):', len(fft_sig_pos))
    print('RFFT[fc]:', fft_sig_pos[fft_index])

    # By scaling by number of points, the RFFT returns unit amplitude for unit input
    # when the frequencies are matched to the FFT frequencies
    fft_sig_pos /= len(fft_sig_pos)
    fft_abs = np.abs(fft_sig_pos)

    # Show the waveform and its FFT over the whole record:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(8, 5))
    ax1.plot(time_nd/frequency_sample_rate_hz, mic_sig)
    ax1.set_title('Synthetic CW, no taper')
    ax1.set_xlabel('Time, s')
    ax1.set_ylabel('Norm')
    ax2.semilogx(frequency_fft_pos_hz, fft_abs)
    ax2.set_title('FFT, f = ' + str(round(frequency_center_fft_hz*100)/100) + ' Hz')
    ax2.set_xlabel('Frequency, hz')
    ax2.set_ylabel('FFT amplitude')
    ax2.grid(True)

    plt.show()

