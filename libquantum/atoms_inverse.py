import numpy as np
from libquantum import scales

"""
Atom Reconstruction - inverse CWT for Dictionary 0. Not for chirps, yet.
"""


def morlet2_reconstruct(band_order_Nth, scale_frequency_center_hz, frequency_sample_rate_hz):

    cycles_M, quality_factor_Q = scales.wavelet_MQ_from_N(band_order_Nth)
    morlet2_scale = cycles_M*frequency_sample_rate_hz/scale_frequency_center_hz/(2. * np.pi)
    reconstruct = np.pi**0.25/2/np.sqrt(morlet2_scale)
    return morlet2_scale, reconstruct


def inv_morlet2_prep(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, frequency_sample_rate_hz):

    cycles_M, quality_factor_Q = scales.wavelet_MQ_from_N(band_order_Nth)
    morlet2_scale, reconstruct = morlet2_reconstruct(band_order_Nth, scale_frequency_center_hz, frequency_sample_rate_hz)
    xtime_shifted = frequency_sample_rate_hz*(time_s-offset_time_s)

    return xtime_shifted, morlet2_scale, cycles_M, reconstruct


def inv_morlet2_real(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, cwt_amp_real,
                     frequency_sample_rate_hz):

    xtime_shifted, xscale, cycles_M, rescale = \
        inv_morlet2_prep(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, frequency_sample_rate_hz)
    wavelet_gauss =  np.exp(-0.5 * (xtime_shifted / xscale) ** 2)
    wavelet_gabor_real = wavelet_gauss * np.cos(cycles_M*(xtime_shifted / xscale))

    # Rescale to Morlet wavelet and take the conjugate for imag
    morlet2_inv_real = cwt_amp_real*wavelet_gabor_real
    morlet2_inv_real *= rescale

    return morlet2_inv_real


def inv_morlet2_imag(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, cwt_amp_imag,
                     frequency_sample_rate_hz):
    # TODO: Explain why pi/2 shift has to be removed!
    xtime_shifted, xscale, cycles_M, rescale = \
        inv_morlet2_prep(band_order_Nth, time_s, offset_time_s, scale_frequency_center_hz, frequency_sample_rate_hz)

    wavelet_gauss =  np.exp(-0.5 * (xtime_shifted / xscale) ** 2)
    wavelet_gabor_imag = wavelet_gauss * np.sin(cycles_M*(xtime_shifted / xscale))

    # Rescale to Morlet wavelet and take the conjugate for imag
    morlet2_inv_imag = -cwt_amp_imag*wavelet_gabor_imag
    morlet2_inv_imag *= rescale

    return morlet2_inv_imag
