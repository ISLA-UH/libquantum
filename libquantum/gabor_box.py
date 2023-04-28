"""
Compute the Gabor box for an input envelope.
Scale relative to the ideal Gabor atom uncertainty.
Include chirp and S transform.

"""

from enum import Enum
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy.integrate import trapezoid, simpson
from libquantum.atoms import chirp_MQG_from_N, chirp_spectrum_centered, \
    chirp_spectrum, chirp_scale_from_order, chirp_centered_4cwt
from libquantum.scales import EPSILON
from redvox.common import date_time_utils as dt


# Trapezoidal integration for centered distribution
def integrate_envelope_simpson(y_axis: np.ndarray,
                                  x_axis: np.ndarray):
    """
    Cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid

    :param x_axis: timestamps corresponding to the data in seconds
    :param y_axis: data to integrate using cumulative trapezoid

    :return: integrated data with the same length as the input
    """

    # TODO: Verify x and y have the same dims
    # Multiply by complex conjugate and take real part.
    y_squared = np.real(y_axis*np.conj(y_axis))
    y_magnitude_squared = simpson(y=y_squared,
                                  x=x_axis)
    y_centroid = x_axis*y_squared
    x_centroid = simpson(y=y_centroid,
                         x=x_axis)
    x_centroid /= y_magnitude_squared

    y_variance = ((x_axis - x_centroid)**2)*y_squared
    x_variance = simpson(y=y_variance,
                         x=x_axis)
    x_variance /= y_magnitude_squared

    return x_centroid, x_variance, y_magnitude_squared


if __name__ == "__main__":

    # Test
    sample_frequency_hz = 100.
    center_frequency_hz = 10.
    nth_order = 3.

    # Need this to nondimensionalize the Gabor box
    scale_s = chirp_scale_from_order(band_order_Nth=nth_order,
                                     scale_frequency_center_hz=center_frequency_hz,
                                     frequency_sample_rate_hz=sample_frequency_hz)

    # Need this to compute the support and build time
    # TODO: Make time dimension invariant
    cycles_M, quality_factor_Q, gamma = chirp_MQG_from_N(band_order_Nth=nth_order)
    duration_s = cycles_M/center_frequency_hz
    duration_points = int(duration_s*sample_frequency_hz)
    time_s = np.arange(duration_points)/sample_frequency_hz

    wavelet_chirp, time_centered_s = chirp_centered_4cwt(band_order_Nth=nth_order,
                                                         sig_or_time=time_s,
                                                         scale_frequency_center_hz=center_frequency_hz,
                                                         frequency_sample_rate_hz=sample_frequency_hz,
                                                         dictionary_type="spect")

    time_scaled = time_centered_s*sample_frequency_hz

    # Integral MUST use scaled time
    x_centroid, x_variance, y_magnitude_squared = \
        integrate_envelope_simpson(x_axis=time_scaled, y_axis=wavelet_chirp)

    print("Time, centroid, variance:", x_centroid, x_variance)
    print("Time, scaled uncertainty:", 2*x_variance/scale_s**2)

    wavelet_frequency = np.arange(0, sample_frequency_hz/2, 0.1/duration_s)
    wavelet_spectrum, frequency_shifted_hz = \
        chirp_spectrum_centered(band_order_Nth=nth_order,
                                scale_frequency_center_hz=center_frequency_hz,
                                frequency_sample_rate_hz=sample_frequency_hz)

    wavelet_spectrum2, frequency_shifted_hz2 = \
        chirp_spectrum(frequency_hz=wavelet_frequency,
                       band_order_Nth=nth_order,
                       offset_time_s=0,
                       scale_frequency_center_hz=center_frequency_hz,
                       frequency_sample_rate_hz=sample_frequency_hz)

    plt.figure()
    plt.plot(time_centered_s, np.real(wavelet_chirp)/np.sqrt(y_magnitude_squared))
    plt.plot(time_centered_s, np.imag(wavelet_chirp)/np.sqrt(y_magnitude_squared))
    plt.plot(time_centered_s, np.sqrt(np.abs(wavelet_chirp*np.conj(wavelet_chirp)/y_magnitude_squared)))

    plt.figure()
    plt.plot(frequency_shifted_hz, np.log10(np.abs(wavelet_spectrum) + EPSILON))
    plt.figure()
    plt.plot(wavelet_frequency, np.log10(np.abs(wavelet_spectrum2) + EPSILON))

    plt.show()
