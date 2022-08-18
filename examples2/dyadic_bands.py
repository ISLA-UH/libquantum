"""
dyadic_bands.py
Compute the complete set of Nth order bands from Nyquist to the averaging frequency

"""
import numpy as np
import matplotlib.pyplot as plt
from libquantum.scales import EPSILON, Slice
print(__doc__)

# CONSTANTS
default_scale_order = 3.
default_scale_base = Slice.G2
default_ref_frequency = Slice.F1


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


def main(frequency_sample_hz,
         frequency_ave_hz,
         scale_order: float = default_scale_base,
         scale_ref_hz: float = default_ref_frequency,
         scale_base: float = default_scale_base):
    """

    :param frequency_sample_hz:
    :param frequency_ave_hz:
    :param scale_order:
    :param scale_ref:
    :param scale_base:
    :return:
    """

    print(scale_order, scale_base, scale_ref_hz)
    print(frequency_ave_hz, frequency_sample_hz)

    # Nyquist limit
    n_over_log2g = scale_order/np.log2(scale_base)
    log2_ref = np.log2(frequency_sample_hz/scale_ref_hz)
    band_nyq = int(np.ceil(n_over_log2g*(1-log2_ref)))

    scale_M = scale_multiplier(scale_order)

    time_ave_log2 = np.ceil(np.log2(scale_M*frequency_sample_hz/frequency_ave_hz))
    band_ave = int(np.floor(n_over_log2g*(time_ave_log2 - np.log2(scale_M) - log2_ref)))

    freq_ave_hz = scale_base**(-band_ave/scale_order)
    freq_nyq_hz = scale_base**(-band_nyq/scale_order)

    print(band_ave, band_nyq)
    bands = np.arange(band_nyq, band_ave)
    freqs = scale_base**(-bands/scale_order)
    print(freq_ave_hz, freq_nyq_hz)
    print(freqs)
    # TODO: VERIFY/TEST


if __name__ == "__main__":
    scale_order0 = 12.
    scale_base0 = Slice.G3
    scale_ref0 = Slice.F1

    frequency_sample_hz = 8000.
    frequency_ave_hz = 10.

    main(frequency_sample_hz, frequency_ave_hz, scale_order0, scale_ref0, scale_base0)
