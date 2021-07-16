"""
This module contains functions for examining doppler shift in signals
"""

import numpy as np
from typing import Tuple


def time_duration(time_vector: np.ndarray) -> float:
    """
    Compute time from max and min

    :param time_vector: array of times in seconds
    :return: duration in seconds
    """
    return np.max(time_vector) - np.min(time_vector)


def time_4d_mx(time_array: np.ndarray, space_dimensions: int) -> np.array:
    """
    Input space and convert to spacetime matrix [time x XYZ]
    i.e. if time is [1, 2, 3, 4] and dimensions is 3
    result is [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]

    :param time_array: array of times in seconds
    :param space_dimensions: integer number of dimensions of space
    :return: time_array arranged vertically space_dimensions number of times
    """
    return np.array([time_array, ] * space_dimensions).transpose()


def space_4d_mx(space_column_vector: np.ndarray, time_number_samples: int) -> np.array:
    """
    Input XYZ space vector and convert to spacetime matrix [time x XYZ]

    :param space_column_vector: an array representing 3 dimensional space
    :param time_number_samples: number of time samples
    :return: spacetime matrix
    """
    return np.array([space_column_vector, ] * time_number_samples)


def hadamard_dot_product_mx(x_mx: np.ndarray, y_mx: np.ndarray) -> np.ndarray:
    """"
    Hadamard columm-wise dot product with summation

    :param x_mx: MxN matrix x, columns represent directional vectors
    :param y_mx: MxN matrix y
    :return: sum of matrix product over columns
    """
    test = np.sum(x_mx * y_mx, 1)
    return test


def range_vector_sr(x_initial_position_vector: np.array,
                    x_final_position_vector: np.array) -> np.array:
    """
    Start to end direction, or source vs receiver

    :param x_initial_position_vector: starting position, 3-element XYZ
    :param x_final_position_vector: ending position, 3-element XYZ
    :return: receiver - source vectors
    """
    return x_final_position_vector - x_initial_position_vector


def range_matrix_sr(x_source_mx: np.ndarray,
                    x_receiver_mx: np.ndarray) -> np.ndarray:
    """
    start to end position, source vs receiver matrix

    :param x_source_mx: starting position, 3-element XYZ columns, t rows
    :param x_receiver_mx: ending position, 3-element XYZ columns, t rows
    :return: receiver - source matrices
    """
    return x_receiver_mx - x_source_mx


def range_hadamard(r_mx: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of range and time (Square root of Hadamard dot product over columns)

    :param r_mx: positional range matrix, 3-element XYZ columns, t rows
    :return: Square root of Hadamard dot product over columns
    """
    return np.sqrt(hadamard_dot_product_mx(r_mx, r_mx))


def range_scalar(x_source_vector: np.array, x_receiver_vector: np.array) -> float:
    """
    Compute the magnitude of the range (square root of the sum of the range vector squared, element-wise)

    :param x_source_vector: starting position, 3-element XYZ
    :param x_receiver_vector: ending position, 3-element XYZ
    :return: magnitude of the range
    """
    range_vector = range_vector_sr(x_source_vector, x_receiver_vector)
    return np.sqrt(np.sum(range_vector * range_vector))


def _get_setup(time_array_s: np.array,
               num_space_dimensions: int,
               source_position_vector_initial_xyz_m: np.array,
               source_position_vector_final_xyz_m: np.array,
               receiver_position_vector_initial_xyz_m: np.array,
               receiver_position_vector_final_xyz_m: np.array) -> Tuple[int, np.array, float, float]:
    """
    Apply a doppler shift on a source moving towards the receiver

    :param time_array_s: array of source times in seconds
    :param num_space_dimensions: integer number of space dimensions. Nominal = 3, should match space position vectors.
    :param source_position_vector_initial_xyz_m: initial source position in space (where it initially started)
    :param source_position_vector_final_xyz_m: final source position in space
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param receiver_position_vector_final_xyz_m: position in space where receiver ends up
    :return: number of time samples, matrix for time, matrix for trajectory range for source and receiver
    """
    num_samples = len(time_array_s)
    temp_mx_s = time_4d_mx(time_array_s, num_space_dimensions)
    source_trajectory_m = range_scalar(source_position_vector_initial_xyz_m, source_position_vector_final_xyz_m)
    receiver_trajectory_m = range_scalar(receiver_position_vector_initial_xyz_m, receiver_position_vector_final_xyz_m)

    return num_samples, temp_mx_s, source_trajectory_m, receiver_trajectory_m


def _get_velocity_mps(speed_mps: float,
                      trajectory_m: float,
                      num_samples: int,
                      position_vector_initial_xyz_m: np.array,
                      position_vector_final_xyz_m: np.array) -> np.array:
    """
    Compute velocity (in meter per second)

    :param speed_mps: object's speed
    :param trajectory_m: magnitude of trajectory in meters
    :param num_samples: number of samples
    :param position_vector_initial_xyz_m: position in space where object initially started
    :param position_vector_final_xyz_m: position in space where object ends up
    :return: velocity in XYZ + time matrix
    """
    if speed_mps > 0:
        velocity_direction = range_vector_sr(position_vector_initial_xyz_m, position_vector_final_xyz_m) / trajectory_m
        velocity_mps = speed_mps * velocity_direction
    else:
        velocity_mps = np.zeros(3)
    velocity_mx_mps = space_4d_mx(velocity_mps, num_samples)
    return velocity_mx_mps


def _get_final_vals(spacetime_matrix: np.array,
                    receiver_velocity_mx_mps: np.array,
                    source_velocity_mx_mps: np.array,
                    time_array_s: np.array,
                    receiver_position_vector_initial_xyz_m: np.array,
                    source_position_vector_initial_xyz_m: np.array,
                    signal_speed_mps: float,
                    object_speed_mps: float,
                    num_dimensions: int,
                    num_samples: int,
                    inverse: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute array of times in seconds, Magnitude of range and time, omega over omega center

    :param spacetime_matrix: spacetime matrix
    :param time_array_s: array of times
    :param receiver_velocity_mx_mps: receiver velocity in meters per second
    :param source_velocity_mx_mps: source velocity in meters per second
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param source_position_vector_initial_xyz_m: position in space where source initially started
    :param signal_speed_mps: signal speed in meters per second
    :param object_speed_mps: object speed in meters per second
    :param num_dimensions: number of dimensions
    :param num_samples: number of samples
    :param inverse: is this an inverse calculation?  Default False
    :return: array of times in seconds, Magnitude of range and time, omega over omega center
    """
    # Initial range vector, source to receiver (sr) at start of segment
    range_vector_sr_initial_m = receiver_position_vector_initial_xyz_m - source_position_vector_initial_xyz_m
    range_initial_mx_m = space_4d_mx(range_vector_sr_initial_m, num_samples)
    # Compute object values and phase
    denom = 1. / (signal_speed_mps**2 - object_speed_mps**2)
    # Dot product contracts to 1D size
    if inverse:
        temp_range_mx_m = range_initial_mx_m + receiver_velocity_mx_mps * spacetime_matrix
        term1 = (signal_speed_mps**2) * time_array_s - hadamard_dot_product_mx(source_velocity_mx_mps, temp_range_mx_m)
    else:
        temp_range_mx_m = range_initial_mx_m - source_velocity_mx_mps * spacetime_matrix
        term1 = (signal_speed_mps**2) * time_array_s + \
            hadamard_dot_product_mx(receiver_velocity_mx_mps, temp_range_mx_m)
    term1 *= denom

    temp_range_m = range_hadamard(temp_range_mx_m)
    term2 = temp_range_m**2 - (time_array_s * signal_speed_mps)**2
    term2 *= denom
    if inverse:
        time_s = term1 - np.sqrt(term1**2 + term2)
        range_mx_m = temp_range_mx_m - source_velocity_mx_mps * time_4d_mx(time_s, num_dimensions)
    else:
        time_s = term1 + np.sqrt(term1**2 + term2)
        range_mx_m = temp_range_mx_m + receiver_velocity_mx_mps * time_4d_mx(time_s, num_dimensions)

    range_time_m = range_hadamard(range_mx_m)
    omega_over_omega_center = \
        (signal_speed_mps - hadamard_dot_product_mx(range_mx_m, receiver_velocity_mx_mps) / range_time_m) / \
        (signal_speed_mps - hadamard_dot_product_mx(range_mx_m, source_velocity_mx_mps) / range_time_m)

    return time_s, range_time_m, omega_over_omega_center


def doppler_forward(tau_source_s: np.array,
                    signal_speed_mps: float,
                    source_speed_mps: float,
                    receiver_speed_mps: float,
                    space_dimensions: int,
                    source_position_vector_initial_xyz_m: np.array,
                    source_position_vector_final_xyz_m: np.array,
                    receiver_position_vector_initial_xyz_m: np.array,
                    receiver_position_vector_final_xyz_m: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a doppler shift on a source moving towards the receiver

    :param tau_source_s: array of source times in seconds
    :param signal_speed_mps: speed of the signal in meters per second
    :param source_speed_mps: speed of the source in meters per second
    :param receiver_speed_mps: speed of the receiver in meters per second
    :param space_dimensions: integer number of dimensions everything is moving in
    :param source_position_vector_initial_xyz_m: position in space where source initially started
    :param source_position_vector_final_xyz_m: position in space where source ends up
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param receiver_position_vector_final_xyz_m: position in space where receiver ends up
    :return: array of receiver times in seconds, Magnitude of range and time, omega over omega center
    """
    tau_number_samples, tau_mx_s, source_trajectory_m, receiver_trajectory_m = \
        _get_setup(tau_source_s, space_dimensions, source_position_vector_initial_xyz_m,
                   source_position_vector_final_xyz_m, receiver_position_vector_initial_xyz_m,
                   receiver_position_vector_final_xyz_m)

    source_velocity_mx_mps = _get_velocity_mps(source_speed_mps, source_trajectory_m, tau_number_samples,
                                               source_position_vector_initial_xyz_m,
                                               source_position_vector_final_xyz_m)

    receiver_velocity_mx_mps = _get_velocity_mps(receiver_speed_mps, receiver_trajectory_m, tau_number_samples,
                                                 receiver_position_vector_initial_xyz_m,
                                                 receiver_position_vector_final_xyz_m)
    # Observation time
    return _get_final_vals(tau_mx_s, receiver_velocity_mx_mps, source_velocity_mx_mps, tau_source_s,
                           receiver_position_vector_initial_xyz_m, source_position_vector_initial_xyz_m,
                           signal_speed_mps, receiver_speed_mps, space_dimensions, tau_number_samples, False)


def image_doppler_forward(tau_source_s: np.array,
                          signal_speed_mps: float,
                          source_speed_mps: float,
                          receiver_speed_mps: float,
                          space_dimensions: int,
                          source_position_vector_initial_xyz_m: np.array,
                          source_position_vector_final_xyz_m: np.array,
                          receiver_position_vector_initial_xyz_m: np.array,
                          receiver_position_vector_final_xyz_m: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a doppler shift on a perceived source moving towards the receiver

    :param tau_source_s: array of source times in seconds
    :param signal_speed_mps: speed of the signal in meters per second
    :param source_speed_mps: speed of the source in meters per second
    :param receiver_speed_mps: speed of the receiver in meters per second
    :param space_dimensions: integer number of dimensions everything is moving in
    :param source_position_vector_initial_xyz_m: position in space where source initially started
    :param source_position_vector_final_xyz_m: position in space where source ends up
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param receiver_position_vector_final_xyz_m: position in space where receiver ends up
    :return: array of receiver times in seconds, Magnitude of range and time, omega over omega center
    """
    # Flip the z-axis - everything else is the same
    image_source_position_vector_initial_xyz_m = source_position_vector_initial_xyz_m*np.array([1., 1., -1.])
    image_source_position_vector_final_xyz_m = source_position_vector_final_xyz_m*np.array([1., 1., -1.])

    return doppler_forward(tau_source_s, signal_speed_mps, source_speed_mps, receiver_speed_mps,
                           space_dimensions, image_source_position_vector_initial_xyz_m,
                           image_source_position_vector_final_xyz_m, receiver_position_vector_initial_xyz_m,
                           receiver_position_vector_final_xyz_m)


def doppler_inverse(inv_time_receiver_s: np.array,
                    signal_speed_mps: float,
                    source_speed_mps: float,
                    receiver_speed_mps: float,
                    space_dimensions: int,
                    source_position_vector_initial_xyz_m: np.array,
                    source_position_vector_final_xyz_m: np.array,
                    receiver_position_vector_initial_xyz_m: np.array,
                    receiver_position_vector_final_xyz_m: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a doppler shift on an inverse

    :param inv_time_receiver_s: array of receiver times in seconds
    :param signal_speed_mps: speed of the signal in meters per second
    :param source_speed_mps: speed of the source in meters per second
    :param receiver_speed_mps: speed of the receiver in meters per second
    :param space_dimensions: integer number of dimensions everything is moving in
    :param source_position_vector_initial_xyz_m: position in space where source initially started
    :param source_position_vector_final_xyz_m: position in space where source ends up
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param receiver_position_vector_final_xyz_m: position in space where receiver ends up
    :return: array of receiver times in seconds, Magnitude of range and time, omega over omega center
    """
    time_number_samples, time_mx_s, source_trajectory_m, receiver_trajectory_m = \
        _get_setup(inv_time_receiver_s, space_dimensions, source_position_vector_initial_xyz_m,
                   source_position_vector_final_xyz_m, receiver_position_vector_initial_xyz_m,
                   receiver_position_vector_final_xyz_m)

    source_velocity_mx_mps = _get_velocity_mps(source_speed_mps, source_trajectory_m, time_number_samples,
                                               source_position_vector_initial_xyz_m,
                                               source_position_vector_final_xyz_m)

    receiver_velocity_mx_mps = _get_velocity_mps(receiver_speed_mps, receiver_trajectory_m, time_number_samples,
                                                 receiver_position_vector_initial_xyz_m,
                                                 receiver_position_vector_final_xyz_m)

    # compute range and terms and find Source tau
    return _get_final_vals(time_mx_s, receiver_velocity_mx_mps, source_velocity_mx_mps, inv_time_receiver_s,
                           receiver_position_vector_initial_xyz_m, source_position_vector_initial_xyz_m,
                           signal_speed_mps, source_speed_mps, space_dimensions, time_number_samples, True)


def image_doppler_inverse(inv_time_receiver_s: np.array,
                          signal_speed_mps: float,
                          source_speed_mps: float,
                          receiver_speed_mps: float,
                          space_dimensions: int,
                          source_position_vector_initial_xyz_m: np.array,
                          source_position_vector_final_xyz_m: np.array,
                          receiver_position_vector_initial_xyz_m: np.array,
                          receiver_position_vector_final_xyz_m: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply a doppler shift on an image inverse

    :param inv_time_receiver_s: array of receiver times in seconds
    :param signal_speed_mps: speed of the signal in meters per second
    :param source_speed_mps: speed of the source in meters per second
    :param receiver_speed_mps: speed of the receiver in meters per second
    :param space_dimensions: integer number of dimensions everything is moving in
    :param source_position_vector_initial_xyz_m: position in space where source initially started
    :param source_position_vector_final_xyz_m: position in space where source ends up
    :param receiver_position_vector_initial_xyz_m: position in space where receiver initially started
    :param receiver_position_vector_final_xyz_m: position in space where receiver ends up
    :return: array of receiver times in seconds, Magnitude of range and time, omega over omega center
    """
    # Flip the z-axis - everything else is the same
    image_source_position_vector_initial_xyz_m = source_position_vector_initial_xyz_m*np.array([1., 1., -1.])
    image_source_position_vector_final_xyz_m = source_position_vector_final_xyz_m*np.array([1., 1., -1.])

    return doppler_inverse(inv_time_receiver_s, signal_speed_mps, source_speed_mps, receiver_speed_mps,
                           space_dimensions, image_source_position_vector_initial_xyz_m,
                           image_source_position_vector_final_xyz_m, receiver_position_vector_initial_xyz_m,
                           receiver_position_vector_final_xyz_m)
