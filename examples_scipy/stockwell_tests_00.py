"""
Stockwell
Compute Stockwell TFR from numpy array/tensor
https://mne.tools/stable/generated/mne.time_frequency.tfr_array_stockwell.html#mne.time_frequency.tfr_array_stockwell

"""

import numpy as np
from libquantum.stockwell_orig import tfr_array_stockwell, calculate_rms_sig_test
from matplotlib import pyplot as plt
print(__doc__)


if __name__ == "__main__":
    n_times = 1024  # Just over 1 second epochs
    sfreq = 1000.0  # Sample frequency

    seed = 42
    rng = np.random.RandomState(seed)
    noise = rng.randn(n_times)/10.

    # Add a 50 Hz sinusoidal burst to the noise and ramp it.
    t = np.arange(n_times, dtype=np.float64) / sfreq
    signal = np.sin(np.pi * 2. * 50. * t)  # 50 Hz sinusoid signal
    signal[np.logical_or(t < 0.45, t > 0.55)] = 0.  # Hard windowing
    on_time = np.logical_and(t >= 0.45, t <= 0.55)
    signal[on_time] *= np.hanning(on_time.sum())  # Ramping
    sig_in = signal+noise
    print("Sig n:", sig_in.shape)

    # Compute strided RMS
    rms_sig_wf, rms_sig_time_s = calculate_rms_sig_test(sig_wf=sig_in, sig_time=t)

    plt.figure()
    plt.plot(t, sig_in)
    plt.title('Input waveform')

    plt.figure()
    plt.plot(rms_sig_time_s, rms_sig_wf)
    plt.title('Input waveform')

    freqs = np.arange(5., 100., 3.)
    vmin, vmax = -3., 3.  # Define our color limits.

    # Stockwell
    fmin, fmax = freqs[[0, -1]]
    [st_power, f, _] = tfr_array_stockwell(data=sig_in, sfreq=sfreq, fmin=fmin, fmax=fmax, width=3.0)
    print("Shape of power:", st_power.shape)
    print("Shape of frequencies:", f.shape)
    plt.figure()
    plt.pcolormesh(st_power, cmap='RdBu_r')

    plt.show()