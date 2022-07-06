import os
import numpy as np
import matplotlib.pyplot as plt
from libquantum import utils
from scipy import interpolate, signal
from libquantum import styx_stx, styx_cwt, styx_fft, scales
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq


input_order = 12.
mic_orig_fs_hz = 800.
mic_decimate = 8

# input_file = "/Users/mgarces/Documents/DATA_2022/DOUBLE_DOUBLE/redvox_data/double_1_1637610021.npz"
# lead_s = 30
# lead_pts = int(lead_s*mic_orig_fs_hz/mic_decimate)
# EVENT_NAME = 'Double 1'
input_file = "/Users/mgarces/Documents/DATA_2022/DOUBLE_DOUBLE/redvox_data/double_2_1637610021.npz"
lead_s = 75
lead_pts = int(lead_s*mic_orig_fs_hz/mic_decimate)
EVENT_NAME = 'Double 2'


if __name__ == "__main__":
    npzfile = np.load(input_file, allow_pickle=True)
    #  print(npzfile.files)
    # exit()
    # ['station_id', 'mic_start_utc', 'mic_mpa', 'mic_time_s',
    # 'accz_cmps2', 'accy_cmps2', 'accx_cmps2', 'accz_time_s']
    station_id = npzfile['station_id']
    mic_start_utc = npzfile['mic_start_utc']
    mic_mpa = npzfile['mic_mpa']
    mic_time_s = npzfile['mic_time_s']
    accz_cmps2 = npzfile['accz_cmps2']
    accy_cmps2 = npzfile['accy_cmps2']
    accx_cmps2 = npzfile['accx_cmps2']
    accz_time_s = npzfile['accz_time_s']

    mic_sample_interval_s = np.mean(np.diff(mic_time_s))
    mic_sample_interval_std_s = np.std(np.diff(mic_time_s))
    accz_sample_interval_s = np.mean(np.diff(accz_time_s))
    accz_sample_interval_std_s = np.std(np.diff(accz_time_s))

    mic_sample_rate_hz = 1/mic_sample_interval_s
    accz_sample_rate_hz = 1/accz_sample_interval_s

    # TODO: evaluate for all channels
    # Resample acc
    acc_new_wf, acc_new_time_s = \
        utils.resample_uneven_signal(sig_wf=accz_cmps2,
                                     sig_epoch_s=accz_time_s,
                                     sample_rate_new_hz=mic_sample_rate_hz)

    print(acc_new_wf.shape)
    print(acc_new_time_s.shape)

    accz_new_sample_interval_s = np.mean(np.diff(acc_new_time_s))
    accz_new_sample_rate_hz = 1/accz_new_sample_interval_s

    print(input_file)
    print('Mic sample interval, s:', mic_sample_interval_s)
    print('Mic sample interval std, s', mic_sample_interval_std_s)
    print('AccZ sample interval, s:', accz_sample_interval_s)
    print('AccZ sample interval std, s', accz_sample_interval_std_s)
    print('Sample Rates in Hz')
    print('Mic sample rate, hz:', mic_sample_rate_hz)
    print('AccZ sample rate, hz:', accz_sample_rate_hz)
    print('AccZ new sample rate, hz:', accz_new_sample_rate_hz)

    # Plot the acceleration data
    fig1, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    ax[0].plot(acc_new_time_s, acc_new_wf)
    ax[1].plot(accz_time_s, accz_cmps2)
    # Set labels and subplot title
    ax[0].set_ylabel('New Acc Z, cm/s2')
    ax[1].set_ylabel('Acc Z, cm/s2')
    ax[1].set_xlabel(f"Seconds from {mic_start_utc} UTC")
    plt.suptitle(f"RedVox ID {station_id}")

    # Plot mic and accel
    fig2, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    ax[0].plot(mic_time_s, mic_mpa)
    ax[1].plot(accz_time_s, accz_cmps2)
    # Set labels and subplot title
    ax[0].set_ylabel('Mic, mPa')
    ax[1].set_ylabel('Acc Z, cm/s2')
    ax[1].set_xlabel(f"Seconds from {mic_start_utc} UTC")
    plt.suptitle(f"RedVox ID {station_id}")

    # Decimate down to 100 Hz
    mic_sig_dec = signal.decimate(x=mic_mpa, q=mic_decimate)
    acc_sig_dec = signal.decimate(x=acc_new_wf, q=mic_decimate)
    sample_rate_dec_hz = mic_sample_rate_hz/mic_decimate

    pow2_points = 2**int(np.log2(len(mic_sig_dec)))
    print('Len sig:', len(mic_sig_dec))
    print('New len sig:', pow2_points)

    mic_sig_pow2 = mic_sig_dec[lead_pts:lead_pts + pow2_points]
    mic_sig_pow2 /= np.max(np.abs(mic_sig_pow2))
    acc_sig_pow2 = acc_sig_dec[lead_pts:lead_pts + pow2_points]
    acc_sig_pow2 /= np.max(np.abs(acc_sig_pow2))
    # time_s = mic_time_s[lead_pts:lead_pts + pow2_points]
    # time_s -= time_s[0]
    time_s = np.arange(len(mic_sig_pow2))/sample_rate_dec_hz

    mic_sig_pow2 *= utils.taper_tukey(sig_wf_or_time=time_s, fraction_cosine=0.2)
    acc_sig_pow2 *= utils.taper_tukey(sig_wf_or_time=time_s, fraction_cosine=0.2)

    # FFT/CWT/STX Display parameters
    nfft = int(len(mic_sig_pow2) / 32)
    # fmin = 0.25
    # fmax = mic_sample_rate_hz/2  # Nyquist
    fmin = 0.5
    fmax = 50

    # Bandpass
    mic_sig = styx_fft.butter_highpass(sig_wf=mic_sig_pow2,
                                       frequency_sample_rate_hz=sample_rate_dec_hz,
                                       frequency_cut_low_hz=fmin,
                                       tukey_alpha=0.1)

    acc_sig = styx_fft.butter_highpass(sig_wf=acc_sig_pow2,
                                       frequency_sample_rate_hz=sample_rate_dec_hz,
                                       frequency_cut_low_hz=fmin,
                                       tukey_alpha=0.1)

    # Atom scales
    order_Nth, cycles_M, frequency_center_geometric, frequency_start, frequency_end = \
        styx_cwt.scale_frequency_bands(scale_order_input=input_order,
                                       frequency_low_input=fmin,
                                       frequency_sample_rate_input=sample_rate_dec_hz,
                                       frequency_high_input=fmax)
    # Flip to match
    frequency_cwt_fft_hz = np.flip(frequency_center_geometric)
    print('Len freq_cwt:', len(frequency_cwt_fft_hz))
    print('Total number of points in CWT/STX TFR:', len(frequency_cwt_fft_hz)*len(mic_sig))

    # Compute TFR for Mic
    frequency_stft_hz, time_stft_s, stft_complex = \
        styx_fft.stft_complex_pow2(sig_wf=mic_sig,
                                   frequency_sample_rate_hz=sample_rate_dec_hz,
                                   nfft_points=nfft)

    stft_power = 2*np.abs(stft_complex)**2
    stft_log2_power = np.log2(stft_power + scales.EPSILON)
    stft_log2_power -= np.max(stft_log2_power)

    # CWT
    frequency_cwt_hz, time_cwt_s, cwt_complex = \
        styx_cwt.cwt_complex_any_scale_pow2(sig_wf=mic_sig,
                                            frequency_sample_rate_hz=sample_rate_dec_hz,
                                            frequency_cwt_hz=frequency_cwt_fft_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    cwt_power = 2*np.abs(cwt_complex)**2
    cwt_log2_power = np.log2(cwt_power + scales.EPSILON)
    cwt_log2_power -= np.max(cwt_log2_power)

    # STX
    frequency_stx_hz, time_stx_s, stx_complex = \
        styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_sig,
                                            frequency_sample_rate_hz=sample_rate_dec_hz,
                                            frequency_stx_hz=frequency_cwt_fft_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    stx_power = 2*np.abs(stx_complex)**2
    stx_log2_power = np.log2(stx_power + scales.EPSILON)
    stx_log2_power -= np.max(stx_log2_power)

    # Plot resampled, normalized mic and accel with 2^n points
    fig3, ax = plt.subplots(nrows=2, ncols=1, sharex='col')
    ax[0].plot(time_s, mic_sig)
    ax[0].set_xlim(time_s[0], time_s[-1])
    ax[0].set_ylabel('Mic')
    ax[0].grid(True)
    ax[1].plot(time_s, acc_sig)
    ax[1].set_xlim(time_s[0], time_s[-1])
    ax[1].set_ylabel('AccZ')
    ax[1].grid(True)
    ax[1].set_xlabel(f"Seconds from {mic_start_utc} UTC")
    plt.suptitle(f"RedVox ID {station_id}")

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=stft_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic STFT for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_cwt_s,
                           mesh_time=time_cwt_s,
                           mesh_frequency=frequency_cwt_hz,
                           mesh_panel_b_tfr=cwt_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic CWT for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stx_s,
                           mesh_frequency=frequency_stx_hz,
                           mesh_panel_b_tfr=stx_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic STX for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    # Compute TFR for Acc
    # STFT
    frequency_stft_hz, time_stft_s, stft_complex = \
        styx_fft.stft_complex_pow2(sig_wf=acc_sig,
                                   frequency_sample_rate_hz=sample_rate_dec_hz,
                                   nfft_points=nfft)

    stft_power = 2*np.abs(stft_complex)**2
    stft_log2_power = np.log2(stft_power + scales.EPSILON)
    stft_log2_power -= np.max(stft_log2_power)

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=acc_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=stft_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="AccZ STFT for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    # # Atom scales for acc
    # # TODO: In this case, accz has been resampled to match  mic. Could extract from mic.
    # order_Nth, _, frequency_center_geometric, _, _ = \
    #     styx_cwt.scale_frequency_bands(scale_order_input=input_order,
    #                                    frequency_low_input=0.6,
    #                                    frequency_sample_rate_input=accz_new_sample_rate_hz,
    #                                    frequency_high_input=70)
    # Flip to match
    frequency_cwt_fft_hz = np.flip(frequency_center_geometric)

    # CWT
    frequency_cwt_hz, time_cwt_s, cwt_complex = \
        styx_cwt.cwt_complex_any_scale_pow2(sig_wf=acc_sig,
                                            frequency_sample_rate_hz=sample_rate_dec_hz,
                                            frequency_cwt_hz=frequency_cwt_fft_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    cwt_power = 2*np.abs(cwt_complex)**2
    cwt_log2_power = np.log2(cwt_power + scales.EPSILON)
    cwt_log2_power -= np.max(cwt_log2_power)

    # STX
    frequency_stx_hz, time_stx_s, stx_complex = \
        styx_stx.stx_complex_any_scale_pow2(sig_wf=acc_sig,
                                            frequency_sample_rate_hz=sample_rate_dec_hz,
                                            frequency_stx_hz=frequency_cwt_fft_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    stx_power = 2*np.abs(stx_complex)**2
    stx_log2_power = np.log2(stx_power + scales.EPSILON)
    stx_log2_power -= np.max(stx_log2_power)

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=acc_sig,
                           wf_panel_a_time=time_cwt_s,
                           mesh_time=time_cwt_s,
                           mesh_frequency=frequency_cwt_hz,
                           mesh_panel_b_tfr=cwt_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="AccZ CWT for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=acc_sig,
                           wf_panel_a_time=time_s,
                           mesh_time=time_stx_s,
                           mesh_frequency=frequency_stx_hz,
                           mesh_panel_b_tfr=stx_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="AccZ STX for " + EVENT_NAME,
                           frequency_hz_ymin=fmin,
                           frequency_hz_ymax=fmax)

    plt.show()
