from typing import List, Tuple
import libquantum.synthetics as synthetics
import libquantum.utils as utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Test against GT pulse
    """

    pseudo_period_s = 1.
    frequency_hz = 1/pseudo_period_s
    fs_hz = 40*frequency_hz

    gt_tau, sig_gt, sig_gt_i, sig_gt_d = \
        synthetics.gt_blast_center_integral_and_derivative(frequency_peak_hz=frequency_hz,
                                                           sample_rate_hz=fs_hz)
    gt_integrated = utils.integrate_cumtrapz(timestamps_s=gt_tau, sensor_wf=sig_gt)
    gt_derivative = utils.derivative_diff(timestamps_s=gt_tau, sensor_wf=sig_gt)

    # gt_derivative = utils.derivative_gradient(timestamps_s=gt_tau, sensor_wf=sig_gt)
    # grad is not bad - smoothens the derivative - but not as accurate a representation as diff for a blast
    # # If wish to remove discontinuity
    # gt_derivative_max_index = np.argmax(gt_derivative)
    # gt_derivative_max = np.max(gt_derivative)
    # gt_derivative[np.argmax(gt_derivative)] = 0

    # Double integral
    gt_from_derivative_int1 = utils.integrate_cumtrapz(timestamps_s=gt_tau, sensor_wf=gt_derivative)
    gt_from_derivative_int2 = utils.integrate_cumtrapz(timestamps_s=gt_tau, sensor_wf=gt_from_derivative_int1)

    # Double derivative
    # gt_from_integral_der1 = derivative_gradient(timestamps_s=gt_tau, sensor_wf=gt_integrated)
    gt_from_integral_der1 = utils.derivative_diff(timestamps_s=gt_tau, sensor_wf=sig_gt_i)
    gt_from_integral_der2 = utils.derivative_diff(timestamps_s=gt_tau, sensor_wf=gt_from_derivative_int1)

    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(8, 6),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    axes[0].plot(gt_tau, gt_integrated)
    axes[0].plot(gt_tau, sig_gt_i, 'o')
    axes[0].set_title("Numerical vs Theoretical ('o')")
    axes[0].set_ylabel('GT integral')
    axes[0].grid(True)
    axes[1].plot(gt_tau, sig_gt)
    axes[1].set_ylabel('GT')
    axes[1].grid(True)
    axes[2].plot(gt_tau, gt_derivative)
    axes[2].plot(gt_tau, sig_gt_d, 'o')
    axes[2].set_ylabel('GT derivative')
    axes[2].grid(True)
    axes[2].set_xlabel('tau - 1')

    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(8, 6),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    axes[0].plot(gt_tau, gt_from_derivative_int2)
    axes[0].plot(gt_tau, sig_gt_i, 'o')
    axes[0].set_title("Single and double integrals, numerical vs theoretical ('o')")
    axes[0].set_ylabel('GT integral 2')
    axes[0].grid(True)
    axes[1].plot(gt_tau, gt_from_derivative_int1)
    axes[1].plot(gt_tau, sig_gt, 'o')
    axes[1].set_ylabel('GT Integral 1')
    axes[1].grid(True)
    axes[2].plot(gt_tau, gt_derivative)
    axes[2].plot(gt_tau, sig_gt_d, 'o')
    axes[2].set_ylabel('GT derivative')
    axes[2].grid(True)
    axes[2].set_xlabel('tau - 1')

    fig_ax_tuple: Tuple[plt.Figure, List[plt.Axes]] = \
        plt.subplots(3, 1,
                     figsize=(8, 6),
                     sharex=True)
    fig: plt.Figure = fig_ax_tuple[0]
    axes: List[plt.Axes] = fig_ax_tuple[1]
    axes[0].plot(gt_tau, gt_integrated)
    axes[0].plot(gt_tau, sig_gt_i, 'o')
    axes[0].set_title("Single and double derivatives, numerical vs theoretical ('o')")
    axes[0].set_ylabel('GT integral')
    axes[0].grid(True)
    axes[1].plot(gt_tau, gt_from_integral_der1)
    axes[1].plot(gt_tau, sig_gt, 'o')
    axes[1].set_ylabel('GT derivative 1')
    axes[1].grid(True)
    axes[2].plot(gt_tau, gt_from_integral_der2)
    axes[2].plot(gt_tau, sig_gt_d, 'o')
    axes[2].set_ylabel('GT derivative 2')
    axes[2].grid(True)
    axes[2].set_xlabel('tau - 1')

    plt.show()
