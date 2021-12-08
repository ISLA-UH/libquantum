import numpy as np
import matplotlib.pyplot as plt
from libquantum import atoms, entropy, scales, utils
from libquantum import blast_pulse as kaboom
import libquantum.plot_templates.plot_time_frequency_reps as pltq

if __name__ == "__main__":
    """
    Performs inverse CWT on GP pulse as in Garces (2021)
    """

    # Set Order
    order_Nth = 3

    sig_sample_rate_hz = 200.
    # Pulse frequency
    sig_peak_frequency_hz = 5  # Nominal 1 ton HE, after Kim et al. 2021
    sig_pseudo_period_s = 1 / sig_peak_frequency_hz

    # GT pulse
    # Number of cycles
    window_cycles = 64
    window_duration_s = window_cycles * sig_pseudo_period_s
    # This would be the time to use 2^n
    time_points = int(window_duration_s * sig_sample_rate_hz)
    time_s = np.arange(time_points) / sig_sample_rate_hz
    time_half_s = np.max(time_s)/2.
    time_shifted_s = time_s - time_half_s
    time_scaled = time_shifted_s * sig_peak_frequency_hz

    # Event signal
    event_reference_time_epoch_s = utils.datetime_now_epoch_s()
    sig_gt = kaboom.gt_blast_period_center(time_center_s=time_shifted_s,
                                           pseudo_period_s=sig_pseudo_period_s)
    sig_duration_s = len(sig_gt) / sig_sample_rate_hz
    sig_epoch_s = event_reference_time_epoch_s + time_s

    # Wavelet support and min frequency for specified signal duration
    max_time_s, min_frequency_hz = scales.from_duration(band_order_Nth=order_Nth,
                                                        sig_duration_s=sig_duration_s)
    print('\nRequest Order N=', order_Nth)
    print('Synthetic signal duration, s:', sig_duration_s)
    print('Lowest frequency in hz that can support this order for this signal duration is ', min_frequency_hz)
    print('Scale with signal duration and to Nyquist, default G2 base re F1')

    # TFR: Compute complex wavelet transform (cwt) by specifying the start and end center frequencies
    # and getting the n-1 band below it.
    cwt_frequency_high_hz = sig_sample_rate_hz/2.  # Nyquist
    cwt_frequency_low_hz = min_frequency_hz  # from duration
    print('Highest requested CWT frequency, Hz:', cwt_frequency_high_hz)
    print('Lowest  requested CWT frequency, Hz:', cwt_frequency_low_hz)

    transform_method = ["fft", "conv", "morlet2"]
    dict_type = "norm"  # Norm or tone
    for xform in transform_method:
        EVENT_NAME = "CWT cwt_type=" + xform
        print(EVENT_NAME)
        sig_cwt, sig_cwt_bits, sig_cwt_time_s, sig_cwt_frequency_hz = \
            atoms.cwt_chirp_complex(sig_wf=sig_gt,
                                    frequency_low_hz=cwt_frequency_low_hz,
                                    frequency_high_hz=cwt_frequency_high_hz,
                                    frequency_sample_rate_hz=sig_sample_rate_hz,
                                    scale_base=scales.Slice.G3,
                                    band_order_Nth=order_Nth,
                                    cwt_type=xform,
                                    dictionary_type=dict_type,
                                    index_shift=0)
        sig_cwt_snr, sig_cwt_snr_bits, sig_cwt_snr_entropy = entropy.snr_mean_max(tfr_coeff_complex=sig_cwt)

        pltq.plot_wf_mesh_mesh_vert(redvox_id="GT Synth",
                                    wf_panel_2_sig=sig_gt,
                                    wf_panel_2_time=sig_epoch_s,
                                    mesh_time=sig_cwt_time_s,
                                    mesh_frequency=sig_cwt_frequency_hz,
                                    mesh_panel_1_trf=sig_cwt_bits,
                                    mesh_panel_1_colormap_scaling="range",
                                    mesh_panel_0_tfr=sig_cwt_snr_entropy,
                                    wf_panel_2_units="Norm",
                                    mesh_panel_1_cbar_units="bits",
                                    mesh_panel_0_cbar_units="eSNR bits",
                                    start_time_epoch=event_reference_time_epoch_s,
                                    figure_title=EVENT_NAME,
                                    frequency_hz_ymin=cwt_frequency_low_hz,
                                    frequency_hz_ymax=cwt_frequency_high_hz)


        plt.show()