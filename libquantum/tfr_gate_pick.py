"""
This module contains functions for frequency picking
"""

import numpy as np
from typing import Tuple, Union


def submesh_from_passband(tfr_mesh_2d: np.ndarray,
                          frequency_1d: np.ndarray,
                          frequency_min: float,
                          frequency_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts only the coefficients in the passband of interest

    :param tfr_mesh_2d: assumes frequency is the first dimension (rows)
    :param frequency_1d: frequency vector, same ordering as mesh columns
    :param frequency_min: minimum frequency for passband
    :param frequency_max: maximum frequency for passband
    :return: np.ndarray with submesh values and np.ndarray with submesh frequencies
    """
    index_frequency_hz_min = np.argmin(np.abs(frequency_1d - frequency_min))
    index_frequency_hz_max = np.argmin(np.abs(frequency_1d - frequency_max))
    tfr_submesh = tfr_mesh_2d[index_frequency_hz_min:index_frequency_hz_max, :]
    tfr_submesh_frequency = frequency_1d[index_frequency_hz_min:index_frequency_hz_max]

    return tfr_submesh, tfr_submesh_frequency


def double_submesh_from_passband(tfr1_mesh_2d: np.ndarray,
                                 tfr2_mesh_2d: np.ndarray,
                                 frequency_1d: np.ndarray,
                                 frequency_min: float,
                                 frequency_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts only the coefficients in the passband of interest for two np.ndarray mesh

    :param tfr1_mesh_2d: assumes frequency is the first dimension (rows)
    :param tfr2_mesh_2d: assumes frequency is the first dimension (rows), same dims as tfr1
    :param frequency_1d: frequency vector, same ordering as mesh columns
    :param frequency_min: minimum frequency for passband
    :param frequency_max: maximum frequency for passband
    :return: np.ndarray with submesh values for tfr1, np.ndarray with submesh values for tfr2, and tfr_submesh_frequency
    """
    index_frequency_hz_min = np.argmin(np.abs(frequency_1d - frequency_min))
    index_frequency_hz_max = np.argmin(np.abs(frequency_1d - frequency_max))
    tfr1_submesh = tfr1_mesh_2d[index_frequency_hz_min:index_frequency_hz_max, :]
    tfr2_submesh = tfr2_mesh_2d[index_frequency_hz_min:index_frequency_hz_max, :]
    tfr_submesh_frequency = frequency_1d[index_frequency_hz_min:index_frequency_hz_max]

    return tfr1_submesh, tfr2_submesh, tfr_submesh_frequency


def mesh_peaks_from_passband(mesh_2d: np.ndarray,
                             frequency_1d: np.ndarray,
                             frequency_min: float,
                             frequency_max: float) -> Tuple[Union[np.ndarray, int, float, complex], np.ndarray,
                                                            np.ndarray]:
    """
    Find frequency peaks in the passband of interest for mesh

    :param mesh_2d: array with mesh
    :param frequency_1d: frequency vector, same ordering as mesh columns
    :param frequency_min:minimum frequency for passband
    :param frequency_max: maximum frequency for passband
    :return: maximum value of mesh, np.ndarray with frequency peaks, and np.ndarray with frequency peaks index
    """
    index_frequency_hz_min_band = np.argmin(np.abs(frequency_1d - frequency_min))
    index_frequency_hz_max_band = np.argmin(np.abs(frequency_1d - frequency_max))
    mesh_max = np.max(mesh_2d[index_frequency_hz_min_band:index_frequency_hz_max_band, :],
                      axis=0)
    index_argmax = np.argmax(mesh_2d[index_frequency_hz_min_band:index_frequency_hz_max_band, :],
                             axis=0)
    frequency_peak = np.empty(index_argmax.shape)
    frequency_index = index_frequency_hz_min_band + index_argmax

    for j in range(len(index_argmax)):
        frequency_peak[j] = frequency_1d[frequency_index[j]]

    return mesh_max, frequency_peak, frequency_index


def mesh_peaks(mesh_2d: np.ndarray,
               frequency_1d: np.ndarray) -> Tuple[Union[np.ndarray, int, float, complex], np.ndarray,
                                                  np.ndarray]:
    """
    Find frequency peaks in mesh

    :param mesh_2d: array with mesh
    :param frequency_1d: frequency vector, same ordering as mesh columns
    :return: maximum value of mesh, np.ndarray with frequency peaks, and np.ndarray with frequency peaks index
    """

    mesh_max = np.max(mesh_2d, axis=0)
    frequency_index = np.argmax(mesh_2d, axis=0)
    frequency_peak = np.empty(frequency_index.shape)

    for j in range(len(frequency_index)):
        frequency_peak[j] = frequency_1d[frequency_index[j]]

    return mesh_max, frequency_peak, frequency_index


def peak_mask(tfr_1d: np.array,
              tfr_1d_min: float,
              tfr_1d_max: float) -> np.ndarray:
    """
    Constructs nan masks, useful for transportability

    :param tfr_1d: array with values
    :param tfr_1d_min: minimum value to start mask
    :param tfr_1d_max: maximum value to end mask
    :return: np.ndarray with NaN mask for min/max values provided
    """
    # Initialize 1D mask
    tfr_gate_nan_mask = np.empty(tfr_1d.shape[0])
    tfr_gate_nan_mask[:] = np.nan

    condition1 = (tfr_1d <= tfr_1d_max)
    condition2 = (tfr_1d >= tfr_1d_min)
    tfr_gate_nan_mask[condition1 & condition2] = 1

    return tfr_gate_nan_mask


def mesh_mask(tfr_2d: np.array,
              tfr_1d_min: float,
              tfr_1d_max: float) -> np.ndarray:
    """
    Uses the mighty np.ma framework, best for matrix-specific transforms

    :param tfr_2d: array with mesh
    :param tfr_1d_min: minimum value to start mask
    :param tfr_1d_max: maximum value to end mask
    :return: np.ndarray with NaN mask for min/max values provided
    """
    mesh_condition1 = tfr_2d < tfr_1d_min
    mesh_condition2 = tfr_2d > tfr_1d_max
    tfr_2d_masked = \
        np.ma.masked_where(mesh_condition1,
                           tfr_2d)
    np.ma.masked_where(mesh_condition2,
                       tfr_2d_masked)

    return tfr_2d_masked
