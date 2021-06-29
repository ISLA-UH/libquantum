import numpy as np
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid
from typing import List, Tuple, Optional
import libquantum.synthetics as synthetics
import matplotlib.pyplot as plt


def integrate_cumtrapz(timestamps_s: np.ndarray,
                       sensor_wf: np.ndarray,
                       initial_value: float = 0) -> np.ndarray:
    """
    cumulative trapazoid integration using scipy.integrate.cumulative_trapezoid

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :param initial_value: the value to add in the initial of the integrated data to match length of input (default is 0)
    :return: integrated data with the same length as the input
    """
    integrated_data = cumulative_trapezoid(x=timestamps_s,
                                           y=sensor_wf,
                                           initial=initial_value)
    return integrated_data


def derivative_gradient(timestamps_s: np.ndarray,
                        sensor_wf: np.ndarray) -> np.ndarray:
    """
    Derivative using gradient

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input
    """
    # derivative_data = np.gradient(sensor_wf)/np.gradient(timestamps_s)
    derivative_data = np.gradient(sensor_wf, timestamps_s)

    return derivative_data


def derivative_diff(timestamps_s: np.ndarray,
                    sensor_wf: np.ndarray) -> np.ndarray:
    """
    Derivative using gradient

    :param timestamps_s: timestamps corresponding to the data in seconds
    :param sensor_wf: data to integrate using cumulative trapezoid
    :return: derivative data with the same length as the input
    """
    # derivative_data = np.gradient(sensor_wf)/np.gradient(timestamps_s)
    derivative_data0 = np.diff(sensor_wf)/np.diff(timestamps_s)
    derivative_data = np.append(derivative_data0, np.zeros(1))

    return derivative_data


if __name__ == "__main__":
    """
    Test against GT pulse
    """

    pseudo_period_s = 1.
    time_pos_s = pseudo_period_s/4.
    gt_time_centered, sig_gt, sig_gt_i, sig_gt_d = \
        synthetics.gt_blast_center_integral_and_derivative(frequency_peak_hz=1/pseudo_period_s,
                                                           sample_rate_hz=1000/pseudo_period_s)

    gt_time = gt_time_centered/time_pos_s
    gt_integrated = integrate_cumtrapz(timestamps_s=gt_time, sensor_wf=sig_gt)
    gt_derivative = derivative_diff(timestamps_s=gt_time, sensor_wf=sig_gt)
    gt_derivative_max_index = np.argmax(gt_derivative)
    gt_derivative_max = np.max(gt_derivative)
    gt_derivative[np.argmax(gt_derivative)] = 0


    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(8, 6),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    axes[0].plot(gt_time, gt_integrated)
    axes[0].plot(gt_time, sig_gt_i, 'o')
    axes[0].set_title("Numerical")
    axes[0].set_ylabel('GT integral')
    axes[0].grid(True)
    axes[1].plot(gt_time, sig_gt)
    axes[1].set_ylabel('GT')
    axes[1].grid(True)
    axes[2].plot(gt_time, gt_derivative)
    axes[2].plot(gt_time, sig_gt_d, 'o')
    axes[2].set_ylabel('GT derivative')
    axes[2].grid(True)

    # fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
    #     plt.subplots(3, 1,
    #                  figsize=(8, 6),
    #                  sharex=True)
    # fig: plt.Figure = fig_ax_tuple[0]
    # axes: List[plt.Axes] = fig_ax_tuple[1]
    # axes[0].plot(gt_time, sig_gt_i)
    # axes[0].set_title("Mathematical")
    # axes[0].set_ylabel('GT integral')
    # axes[0].grid(True)
    # axes[1].plot(gt_time, sig_gt)
    # axes[1].set_ylabel('GT')
    # axes[1].grid(True)
    # axes[2].plot(gt_time, sig_gt_d)
    # axes[2].set_ylabel('GT derivative')
    # axes[2].grid(True)

    plt.show()
