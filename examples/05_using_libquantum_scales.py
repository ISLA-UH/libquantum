"""
libquanutm example 5: Example using fractional octave scales and export modules
"""

from libquantum import export, scales

if __name__ == "__main__":
    """
    In this example: Planck band, Scale Period up to Nyquist, Scale Frequency up to Nyquist (G3),
    Scale Frequency up to Nyquist (G2), and Equal tempered scale re A4 = 440 Hz
    """

    print('First Planck band, with minimum pseudo-period bandwidth = ', scales.Slice.T0S)
    for scale_order in [0.75, 1, 1.5, 3, 6, 12]:
        planck_center, planck_low, planck_high, quality_Q = scales.planck_scale_s(scale_order=scale_order)
        print('\nOrder_N:', scale_order)
        print('Number of Oscillations, Q:', quality_Q)
        print('%1s%12s%12s' % ('Time_s Center', 'Lower', 'Upper'))
        print('%12.4e%12.4e%12.4e' % (planck_center, planck_low, planck_high))
        print('Scaled bandwidth:', (planck_high-planck_low)/scales.Slice.T0S)

    print('\n Scale Period up to Nyquist')
    export.print_scales_to_screen(scale_order_input=1,
                                  scale_base_input=scales.Slice.G3,
                                  scale_ref_input=scales.Slice.T0S,
                                  scale_sample_interval_input=1E-43,
                                  scale_high_input=1E-41)

    print('\n Scale Frequency up to Nyquist, G3')
    export.print_frequencies_to_screen(frequency_order_input=1,
                                       frequency_base_input=scales.Slice.G3,
                                       frequency_ref_input=scales.Slice.F1,
                                       frequency_low_input=1/1E-41,
                                       frequency_sample_rate_input=1/1E-43)

    print('\n Scale Frequency up to Nyquist, G2')
    export.print_frequencies_to_screen(frequency_order_input=1,
                                       frequency_base_input=scales.Slice.G2,
                                       frequency_ref_input=scales.Slice.F1,
                                       frequency_low_input=1/1E-41,
                                       frequency_sample_rate_input=1/1E-43)

    print('\n *** Equal tempered scale re A4 = 440 Hz')
    export.print_frequencies_to_screen(frequency_order_input=12,
                                       frequency_base_input=scales.Slice.G2,
                                       frequency_ref_input=440,
                                       frequency_low_input=16.35,
                                       frequency_sample_rate_input=16000)
