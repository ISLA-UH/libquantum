"""
cyber_dyadic.py
Compute the complete set of Nth order bands from Nyquist to the averaging frequency
Physical to cyber conversion with preferred quasi-dyadic orders
Example: base10, base2, dB, bits

"""

import numpy as np
from libquantum.scales import Slice
print(__doc__)

# CONSTANTS
default_scale_order = 3.
default_scale_base = Slice.G2
default_ref_frequency = Slice.F1
default_ref_period = Slice.T1S


def scale_multiplier(scale_order: float = default_scale_order):
    """
    Scale multiplier for scale bands of order N > 0.75
    :param scale_order: scale order
    :return:
    """
    return 0.75*np.pi*scale_order


def base_multiplier(scale_order: float = default_scale_order,
                    scale_base: float = default_scale_base):
    """
    Dyadic (log2) foundation for arbitrary base
    :param scale_order:
    :param scale_base:
    :return:
    """
    return scale_order/np.log2(scale_base)


def scale_bands_from_ave(frequency_sample_hz: float,
                         frequency_ave_hz: float,
                         scale_order: float = default_scale_base,
                         scale_ref_hz: float = default_ref_frequency,
                         scale_base: float = default_scale_base):
    """

    :param frequency_sample_hz:
    :param frequency_ave_hz:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: [band_nyq, band_ave, band_max, log2_ave_life_dyad]
    """

    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on reference and averaging frequency
    log2_ave_physical = np.log2(scale_ref_hz/frequency_ave_hz)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)
    log2_ave_cyber = np.log2(frequency_sample_hz/frequency_ave_hz)

    # Closest largest power of two to averaging frequency
    log2_ave_life_dyad = np.ceil(log2_scale_mult + log2_ave_cyber)

    # Shortest period, Nyquist limit
    band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Closest period band
    band_ave = int(np.round(order_over_log2base*log2_ave_physical))
    # Longest period band
    band_max = int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    return [band_nyq, band_ave, band_max, log2_ave_life_dyad]


def scale_bands_from_pow2(frequency_sample_hz: float,
                          log2_ave_life_dyad: int,
                          scale_order: float = default_scale_base,
                          scale_ref_hz: float = default_ref_frequency,
                          scale_base: float = default_scale_base):
    """

    :param frequency_sample_hz:
    :param log2_ave_life_dyad:
    :param scale_order:
    :param scale_ref_hz:
    :param scale_base:
    :return: [band_nyq, band_max]
    """

    # Framework constants
    scale_mult = scale_multiplier(scale_order)
    order_over_log2base = base_multiplier(scale_order, scale_base)

    # Dyadic transforms
    log2_scale_mult = np.log2(scale_mult)

    # Dependent on sensor sample rate
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)

    # Shortest period, Nyquist limit
    band_nyq = int(np.ceil(order_over_log2base*(1-log2_ref)))
    # Longest period band
    band_max = int(np.floor(order_over_log2base*(log2_ave_life_dyad - log2_scale_mult - log2_ref)))

    return [band_nyq, band_max]


def period_from_band(band_min: int,
                        band_max: int,
                        scale_order: float = default_scale_base,
                        scale_ref_s: float = default_ref_period,
                        scale_base: float = default_scale_base):

    bands = np.arange(band_min, band_max+1)
    # Increasing order
    period = scale_ref_s*scale_base**(bands/scale_order)
    return period


def frequency_from_band(band_min: int,
                        band_max: int,
                        scale_order: float = default_scale_base,
                        scale_ref_hz: float = default_ref_frequency,
                        scale_base: float = default_scale_base):

    bands = np.arange(band_min, band_max+1)
    # Flip so it increases
    frequency = np.flip(scale_ref_hz*scale_base**(-bands/scale_order))
    return frequency


if __name__ == "__main__":
    # Framework specs
    scale_order0 = 6.
    scale_base0 = Slice.G3
    scale_ref0 = Slice.F1

    # TFR Lower Limit, may want to lock
    frequency_ave_hz0 = 10.

    # Sensor specific
    frequency_sample_hz0 = 48000.

    print(scale_order0, scale_base0, scale_ref0)
    print(frequency_ave_hz0, frequency_sample_hz0)

    [band_nyq, band_ave, band_max, log2_ave_life_dyad0] = \
        scale_bands_from_ave(frequency_sample_hz0, frequency_ave_hz0, scale_order0, scale_ref0, scale_base0)

    print(band_nyq, band_ave, band_max)
    # Min, Max, and Spec frequency
    freq_min_hz = scale_base0**(-band_max/scale_order0)
    freq_ave_hz = scale_base0**(-band_ave/scale_order0)
    freq_nyq_hz = scale_base0**(-band_nyq/scale_order0)
    print(freq_min_hz, freq_ave_hz, freq_nyq_hz)

    # Reproduce
    [band_nyq, band_max] = scale_bands_from_pow2(frequency_sample_hz0, log2_ave_life_dyad0, scale_order0, scale_ref0, scale_base0)
    freqs = frequency_from_band(band_nyq, band_max, scale_order0, scale_ref0, scale_base0)
    print('Physical freq:', freqs)
    print('Number of bands:', len(freqs))



