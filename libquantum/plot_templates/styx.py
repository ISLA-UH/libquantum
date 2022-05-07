import numpy as np
from scipy.fft import fft, rfft, ifft, fftfreq, fftshift
from libquantum.scales import EPSILON
# TODO: Construct Styx repo


def tfr_stockwell(data, sample_rate, fmin=None, fmax=None, n_fft=None, df=None,
                             order=8.0, binary_order: bool = False):

    M = 12/5*order
    sample_interval = 1/sample_rate
    if n_fft is None:
        n_fft = data.shape[-1]

    # TODO: Add zero padding
    Tw = n_fft/sample_rate
    fmin_nth = 12/5 * order/Tw

    if fmin is None:
        fmin = fmin_nth
    if fmax is None:
        fmax = sample_rate/2.
    if df is None:
        # TODO: Optimize
        df = (fmax-fmin)/12.

    # Take FFT and concatenate
    X = fft(data)
    XX = np.concatenate([X, X], axis=-1)
    frequency_fft = fftfreq(n_fft, sample_interval)
    mu_fft = 2*np.pi*frequency_fft/sample_rate

    # TODO: Standardize
    start_f_idx = np.abs(frequency_fft - fmin).argmin()
    stop_f_idx = np.abs(frequency_fft - fmax).argmin()
    f_start = frequency_fft[start_f_idx]
    f_stop = frequency_fft[stop_f_idx]
    if binary_order is True:
        num_octaves = np.log2(f_stop/f_start)
        num_bands = int(num_octaves*order)
        print("Number of bands:", num_bands)
        frequency_stx = np.logspace(np.log2(f_start), np.log2(f_stop), num=num_bands, base=2.)
    else:
        frequency_stx = np.arange(f_start, f_stop, df)
    print("Shape of frequency_stx", frequency_stx.shape)
    print("SX Band edges:", frequency_stx[0], frequency_stx[-1])

    # Construct shifting frequency indexes
    frequency_stx_index = np.empty(len(frequency_stx), dtype=int)
    frequency_stx_fft = np.empty(len(frequency_stx))

    # Construct time domain and fft of window
    windows_tdr = np.empty((len(frequency_stx), n_fft))
    windows_fft = np.empty((len(frequency_stx), n_fft), dtype=np.complex128)

    tfr_stx = np.empty((len(frequency_stx), n_fft), dtype=np.complex128)
    nu_s = 2*np.pi*frequency_stx/sample_rate

    for isx, fsx in enumerate(frequency_stx):
        stx_index = np.abs(frequency_fft - fsx).argmin()
        fsx_fft = frequency_fft[stx_index[isx]]
        nu_sx = 2*np.pi*fsx/sample_rate

    # TODO: find fft frequencies that exactly match - necessary for shifting FFT
    # Gabor window
    # TODO: Reconcile with above granularity
    nu_s = 2*np.pi*frequency_stx/sample_rate  # Nondimensionalized angular frequency

    # TODO: These can be constructed as a matrix w/o for loop
    for i_f, nu in enumerate(nu_s):
        if nu == 0.:
            windows_fft[i_f] = np.ones(len(data))
        else:
            sigma = M/nu
            windows_fft[i_f] = np.exp(-0.5 * (sigma ** 2.) * (mu_fft ** 2.))

    psd3 = np.empty((len(windows_fft), n_fft))
    # Frequency shift the FFT, then compute the IFFT
    for i_f, window in enumerate(windows_fft):
        # this is f_fft + f_s
        f_idx = frequency_stx_index[i_f]
        TFR = ifft(XX[f_idx:f_idx + n_fft] * window) + EPSILON
        TFR_abs = np.abs(TFR)
        TFR_abs *= TFR_abs  # Square
        psd3[i_f, :] = TFR_abs
    print("W3 shape:", windows_fft.shape)
    return psd3, frequency_stx, frequency_fft, windows_fft
