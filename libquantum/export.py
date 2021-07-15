"""
This module contains functions for exporting time scales and frequencies to screen
"""

from libquantum import scales


def print_scales_to_screen(scale_order_input: float,
                           scale_base_input: float,
                           scale_ref_input: float,
                           scale_sample_interval_input: float,
                           scale_high_input: float) -> None:
    """
    Displays time scales to screen for examination and presentation

    :param scale_order_input: Nth order specification
    :param scale_base_input: G2 or G3
    :param scale_ref_input: Reference time scale
    :param scale_sample_interval_input: inverse of the sample rate
    :param scale_high_input: highest scale of interest
    :return: plot to screen
    """

    scale_order, scale_base, scale_band_number, scale_ref, scale_center_algebraic, scale_center_geometric, \
    scale_start, scale_end = \
        scales.band_periods_nyquist(scale_order_input, scale_base_input,
                                    scale_ref_input,
                                    scale_sample_interval_input, scale_high_input)

    print('Scale Order:', scale_order)
    print('Reference scale:', scale_ref)
    print('%8s%12s%12s%12s%12s' % ('Band #', 'Center Geo', 'Center Alg', 'Lower', 'Upper'))
    for i in range(scale_band_number.size):
        print('%+8i%12.4e%12.4e%12.4e%12.4e' % (int(scale_band_number[i]),
                                                scale_center_geometric[i], scale_center_algebraic[i],
                                                scale_start[i], scale_end[i]))


def print_frequencies_to_screen(frequency_order_input: float,
                                frequency_base_input: float,
                                frequency_ref_input: float,
                                frequency_low_input: float,
                                frequency_sample_rate_input: float) -> None:
    """
    Displays frequencies to screen for examination and presentation

    :param frequency_order_input: Nth order
    :param frequency_base_input: G2 or G3
    :param frequency_ref_input: reference frequency
    :param frequency_low_input: lowest frequency of interest
    :param frequency_sample_rate_input: sample rate
    :return: plot to screen
    """
    frequency_order, frequency_base, frequency_band_number, frequency_ref, frequency_center_algebraic, \
    frequency_center_geometric, frequency_start, frequency_end = \
        scales.band_frequencies_nyquist(frequency_order_input, frequency_base_input,
                                        frequency_ref_input,
                                        frequency_low_input, frequency_sample_rate_input)

    print('Scale Order:', frequency_order)
    print('Reference frequency:', frequency_ref)
    print('%8s%12s%12s%12s%12s' % ('Band #', 'Center Geo', 'Center Alg', 'Lower', 'Upper'))
    for i in range(frequency_band_number.size):
        print('%+8i%12.4e%12.4e%12.4e%12.4e' % (int(frequency_band_number[i]),
                                                frequency_center_geometric[i], frequency_center_algebraic[i],
                                                frequency_start[i], frequency_end[i]))
