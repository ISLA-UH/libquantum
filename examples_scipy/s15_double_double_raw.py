import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from libquantum import utils
from scipy import interpolate, signal
from libquantum import styx_stx, styx_cwt, styx_fft, scales
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq


# input_file = "/Users/mgarces/Documents/DATA_2022/DOUBLE_DOUBLE/redvox_data/double_1_1637610021.npz"
# lead_s = 30 + 50
# EVENT_NAME = 'Double 1'

input_file = "/Users/mgarces/Documents/DATA_2022/DOUBLE_DOUBLE/redvox_data/double_2_1637610021.npz"
lead_s = 75 + 45
EVENT_NAME = 'Double 2'

# Gabor atom averaging spec (lowest frequency)
input_order = 12.
frequency_averaging_hz = 0.5
number_cycles_averaging = 3*np.pi/4 * input_order
duration_averaging_s = number_cycles_averaging / frequency_averaging_hz

# Targeted Duration
duration_s = 60

# Bandwidth spec - build override
frequency_cutoff_low_hz = frequency_averaging_hz
frequency_cutoff_high_hz = 100

# Display spec
pixels_per_mesh = 2**19


def plt_two_time_series(time_1, sig_1, time_2, sig_2,
                        y_label1, y_label2,
                        title_label,
                        sta_id,
                        datetime_start_utc,
                        x_label: str = "UTC Seconds from"):
    """
    Quick plots
    """
    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0].plot(time_1, sig_1)
    ax[1].plot(time_2, sig_2)
    # Set labels and subplot title
    ax[0].set_ylabel(y_label1)
    ax[1].set_ylabel(y_label2)
    ax[1].set_xlabel(x_label + " " + str(datetime_start_utc))
    plt.suptitle(title_label + " " + str(sta_id))


if __name__ == "__main__":
    # Load file
    npzfile = np.load(input_file, allow_pickle=True)
    #  print(npzfile.files)
    # exit()
    # ['station_id', 'mic_start_utc', 'mic_mpa', 'mic_time_s',
    # 'accz_cmps2', 'accy_cmps2', 'accx_cmps2', 'accz_time_s']
    station_id = npzfile['station_id']
    mic_start_utc = npzfile['mic_start_utc']

    mic_time_s = npzfile['mic_time_s']
    mic_mpa = npzfile['mic_mpa']
    acc_time_s = npzfile['accz_time_s']
    accz_cmps2 = npzfile['accz_cmps2']
    accy_cmps2 = npzfile['accy_cmps2']
    accx_cmps2 = npzfile['accx_cmps2']

    # THE NEXT STEPS ARE JUST PREP:
    # Compute sample rates and number of points
    mic_fs_hz: float = 1/np.mean(np.diff(mic_time_s))  # 800.
    mic_duration_points: int = int(np.ceil(duration_s*mic_fs_hz))

    # Select the specified window
    lead_pts = int(lead_s*mic_fs_hz)
    mic_time_start = mic_time_s[lead_pts]
    mic_time_stop = mic_time_s[lead_pts+mic_duration_points]
    acc_start_idx = np.argmin(np.abs(acc_time_s-mic_time_start))
    acc_stop_idx = np.argmin(np.abs(acc_time_s-mic_time_stop))
    acc_time_start = acc_time_s[acc_start_idx]
    acc_time_stop = acc_time_s[acc_stop_idx]
    # print(mic_time_start)
    # print(acc_time_start)
    # print(mic_time_stop)
    # print(acc_time_stop)

    # Correct time lines
    start_utc = mic_start_utc + timedelta(seconds=mic_time_start)
    print('\nFile name:', input_file)
    print('Original start time:', mic_time_start)
    print('Signal start time:', start_utc)
    mic_t = np.copy(mic_time_s[lead_pts:lead_pts+mic_duration_points])
    mic_t -= mic_time_s[lead_pts]
    acc_t = np.copy(acc_time_s[acc_start_idx:acc_stop_idx])
    acc_t -= mic_time_s[lead_pts]

    # Waveforms
    mic_w = np.copy(mic_mpa[lead_pts:lead_pts+mic_duration_points])
    acc_z = np.copy(accz_cmps2[acc_start_idx:acc_stop_idx])
    acc_y = np.copy(accy_cmps2[acc_start_idx:acc_stop_idx])
    acc_x = np.copy(accx_cmps2[acc_start_idx:acc_stop_idx])

    """
    THIS IS THE REAL BEGINNING OF THE METHODS
    """
    # These are the selected signals with 60s durations
    # They will require zero padding to meet 2^N FFT spec, and then truncating to display

    # Collect fundamental metrics
    mic_sample_interval_s = np.mean(np.diff(mic_t))
    mic_sample_interval_std_s = np.std(np.diff(mic_t))
    mic_sample_rate_hz = 1/mic_sample_interval_s
    mic_points: int = len(mic_w)
    mic_points_pow2: int = 2**int(np.ceil(np.log2(mic_points)))
    mic_points_pow2_atom: int = 2**int(np.ceil(np.log2(duration_averaging_s*mic_sample_rate_hz)))

    acc_sample_interval_s = np.mean(np.diff(acc_t))
    acc_sample_interval_std_s = np.std(np.diff(acc_t))
    acc_sample_rate_hz = 1/acc_sample_interval_s
    acc_points: int = len(acc_z)
    acc_points_pow2: int = 2**int(np.ceil(np.log2(acc_points)))
    acc_points_pow2_atom: int = 2**int(np.ceil(np.log2(duration_averaging_s*acc_sample_rate_hz)))

    # *** Begin processing chain ***
    # Highpass, could be a simpler representation
    fraction_cosine = 0.1
    mic_sig = styx_fft.butter_highpass(sig_wf=mic_w,
                                       frequency_sample_rate_hz=mic_sample_rate_hz,
                                       frequency_cut_low_hz=frequency_cutoff_low_hz,
                                       tukey_alpha=fraction_cosine)

    accz_sig = styx_fft.butter_highpass(sig_wf=acc_z,
                                        frequency_sample_rate_hz=acc_sample_rate_hz,
                                        frequency_cut_low_hz=frequency_cutoff_low_hz,
                                        tukey_alpha=fraction_cosine)

    print('Mic sample interval, s:', mic_sample_interval_s)
    print('Mic sample interval std, s:', mic_sample_interval_std_s)
    print('AccZ sample interval, s:', acc_sample_interval_s)
    print('AccZ sample interval std, s:', acc_sample_interval_std_s)
    print('\nSample Rates in Hz')
    print('Mic sample rate, hz:', mic_sample_rate_hz)
    print('Acc sample rate, hz:', acc_sample_rate_hz)

    # Reconcile Order and Duration specs
    print('\nReconcile with 2^N FFT req for Nth order')
    print('mic_points:', mic_points)
    print('mic_points_pow2:', mic_points_pow2)
    print('mic_points_pow2_atom:', mic_points_pow2_atom)
    print('acc_points:', acc_points)
    print('acc_points_pow2:', acc_points_pow2)
    print('acc_points_pow2_atom:', acc_points_pow2_atom)

    # Zero pad relative to the largest of points_pow2 or points_pow2_atom
    if mic_points_pow2_atom > mic_points_pow2:
        mic_pad_points = mic_points_pow2_atom - mic_points
        # TODO: Study reflect_type
        mic_pad_sig = np.pad(array=mic_sig, pad_width=(mic_pad_points, 0))
    else:
        # TODO: Verify
        mic_pad_points = mic_points_pow2 - mic_points
        mic_pad_sig = np.pad(array=mic_sig, pad_width=(mic_pad_points, 0))

    if acc_points_pow2_atom > acc_points_pow2:
        acc_pad_points = acc_points_pow2_atom - acc_points
        # TODO: Study reflect_type
        acc_pad_sig = np.pad(array=accz_sig, pad_width=(acc_pad_points, 0))
    else:
        acc_pad_points = acc_points_pow2 - acc_points
        acc_pad_sig = np.pad(array=accz_sig, pad_width=(acc_pad_points, 0))

    mic_pad_time = np.arange(len(mic_pad_sig))/mic_sample_rate_hz
    acc_pad_time = np.arange(len(acc_pad_sig))/acc_sample_rate_hz

    # TDR PLOTS
    # Plot raw mic and zoomed signal
    plt_two_time_series(time_1=mic_time_s, sig_1=mic_mpa,
                        time_2=mic_t+lead_s, sig_2=mic_w,
                        y_label1='Mic, mPa',
                        y_label2='Mic, mPa, zoom in',
                        title_label='Record and Selected Sig for RedVox ID',
                        sta_id=station_id,
                        datetime_start_utc=mic_start_utc)

    # Plot raw mic and accel
    plt_two_time_series(time_1=mic_t, sig_1=mic_w,
                        time_2=acc_t, sig_2=acc_z,
                        y_label1='Mic, mPa',
                        y_label2='Acc Z, cm/s2',
                        title_label='Raw Mic and AccZ Sig for RedVox ID',
                        sta_id=station_id,
                        datetime_start_utc=start_utc)

    # Plot highpass mic and accel
    plt_two_time_series(time_1=mic_t, sig_1=mic_sig,
                        time_2=acc_t, sig_2=accz_sig,
                        y_label1='Mic, mPa',
                        y_label2='Acc Z, cm/s2',
                        title_label='Higpassed Mic and AccZ Sig for RedVox ID',
                        sta_id=station_id,
                        datetime_start_utc=start_utc)

    # Plot padded mic and accel
    plt_two_time_series(time_1=np.arange(len(mic_pad_sig)), sig_1=mic_pad_sig,
                        time_2=np.arange(len(acc_pad_sig)), sig_2=acc_pad_sig,
                        y_label1='Mic, mPa',
                        y_label2='Acc Z, cm/s2',
                        title_label='Padded Mic and AccZ Sig for RedVox ID',
                        sta_id=station_id,
                        datetime_start_utc=start_utc,
                        x_label='FFT Data points')

    # Mic atom scales
    if frequency_cutoff_high_hz > mic_sample_rate_hz/2:
        mic_frequency_cutoff_high_hz = mic_sample_rate_hz/2
    else:
        mic_frequency_cutoff_high_hz = np.copy(frequency_cutoff_high_hz)

    order_Nth, cycles_M, frequency_center_geometric, frequency_start, frequency_end = \
        styx_cwt.scale_frequency_bands(scale_order_input=input_order,
                                       frequency_sample_rate_input=mic_sample_rate_hz,
                                       frequency_low_input=frequency_cutoff_low_hz,
                                       frequency_high_input=frequency_cutoff_high_hz)
    # Flip to match STFT
    frequency_inferno_hz = np.flip(frequency_center_geometric)
    print('\nLen Inferno Frequency Bands:', len(frequency_inferno_hz))
    # Compute TFRs for Mic
    nfft = int(len(mic_pad_sig)/16)
    frequency_stft_hz, time_stft_s, stft_complex = \
        styx_fft.stft_complex_pow2(sig_wf=mic_pad_sig,
                                   frequency_sample_rate_hz=mic_sample_rate_hz,
                                   nfft_points=nfft)

    stft_power = 2*np.abs(stft_complex)**2
    stft_log2_power = np.log2(stft_power + scales.EPSILON)
    stft_log2_power -= np.max(stft_log2_power)

    # CWT
    frequency_cwt_hz, time_cwt_s, cwt_complex = \
        styx_cwt.cwt_complex_any_scale_pow2(sig_wf=mic_pad_sig,
                                            frequency_sample_rate_hz=mic_sample_rate_hz,
                                            frequency_cwt_hz=frequency_inferno_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    cwt_power = 2*np.abs(cwt_complex)**2
    cwt_log2_power = np.log2(cwt_power + scales.EPSILON)
    cwt_log2_power -= np.max(cwt_log2_power)

    # STX
    frequency_stx_hz, time_stx_s, stx_complex = \
        styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_pad_sig,
                                            frequency_sample_rate_hz=mic_sample_rate_hz,
                                            frequency_stx_hz=frequency_inferno_hz,
                                            band_order_Nth=order_Nth,
                                            dictionary_type="spect")

    stx_power = 2*np.abs(stx_complex)**2
    stx_log2_power = np.log2(stx_power + scales.EPSILON)
    stx_log2_power -= np.max(stx_log2_power)

    print('\nTotal number of points in STFT TFR:', stft_power.size)
    print('Total number of points in CWT TFR:', cwt_power.size)
    print('Total number of points in STX TFR:', stx_power.size)
    # print('STX TFR decimation factor:', cwt_power.size)

    exit()

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_pad_sig,
                           wf_panel_a_time=mic_pad_time,
                           mesh_time=time_stft_s,
                           mesh_frequency=frequency_stft_hz,
                           mesh_panel_b_tfr=stft_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic STFT for " + EVENT_NAME,
                           frequency_hz_ymin=frequency_inferno_hz[-0],
                           frequency_hz_ymax=frequency_inferno_hz[-1])

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_pad_sig,
                           wf_panel_a_time=mic_pad_time,
                           mesh_time=time_cwt_s,
                           mesh_frequency=frequency_cwt_hz,
                           mesh_panel_b_tfr=cwt_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic CWT for " + EVENT_NAME,
                           frequency_hz_ymin=frequency_inferno_hz[-0],
                           frequency_hz_ymax=frequency_inferno_hz[-1])

    pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
                           wf_panel_a_sig=mic_pad_sig,
                           wf_panel_a_time=mic_pad_time,
                           mesh_time=time_stx_s,
                           mesh_frequency=frequency_stx_hz,
                           mesh_panel_b_tfr=stx_log2_power,
                           mesh_panel_b_colormap_scaling="range",
                           wf_panel_a_units="Norm",
                           mesh_panel_b_cbar_units="bits",
                           start_time_epoch=0,
                           figure_title="Mic STX for " + EVENT_NAME,
                           frequency_hz_ymin=frequency_inferno_hz[-0],
                           frequency_hz_ymax=frequency_inferno_hz[-1])

    plt.show()
    #
    # # Compute TFR for Acc
    # # STFT
    # frequency_stft_hz, time_stft_s, stft_complex = \
    #     styx_fft.stft_complex_pow2(sig_wf=acc_sig,
    #                                frequency_sample_rate_hz=sample_rate_dec_hz,
    #                                nfft_points=nfft)
    #
    # stft_power = 2*np.abs(stft_complex)**2
    # stft_log2_power = np.log2(stft_power + scales.EPSILON)
    # stft_log2_power -= np.max(stft_log2_power)
    #
    # pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
    #                        wf_panel_a_sig=acc_sig,
    #                        wf_panel_a_time=time_s,
    #                        mesh_time=time_stft_s,
    #                        mesh_frequency=frequency_stft_hz,
    #                        mesh_panel_b_tfr=stft_log2_power,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="AccZ STFT for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)
    #
    # # # Atom scales for acc
    # # # TODO: In this case, accz has been resampled to match  mic. Could extract from mic.
    # # order_Nth, _, frequency_center_geometric, _, _ = \
    # #     styx_cwt.scale_frequency_bands(scale_order_input=input_order,
    # #                                    frequency_low_input=0.6,
    # #                                    frequency_sample_rate_input=accz_new_sample_rate_hz,
    # #                                    frequency_high_input=70)
    # # Flip to match
    # frequency_inferno_hz = np.flip(frequency_center_geometric)
    #
    # # CWT
    # frequency_cwt_hz, time_cwt_s, cwt_complex = \
    #     styx_cwt.cwt_complex_any_scale_pow2(sig_wf=acc_sig,
    #                                         frequency_sample_rate_hz=sample_rate_dec_hz,
    #                                         frequency_cwt_hz=frequency_inferno_hz,
    #                                         band_order_Nth=order_Nth,
    #                                         dictionary_type="spect")
    #
    # cwt_power = 2*np.abs(cwt_complex)**2
    # cwt_log2_power = np.log2(cwt_power + scales.EPSILON)
    # cwt_log2_power -= np.max(cwt_log2_power)
    #
    # # STX
    # frequency_stx_hz, time_stx_s, stx_complex = \
    #     styx_stx.stx_complex_any_scale_pow2(sig_wf=acc_sig,
    #                                         frequency_sample_rate_hz=sample_rate_dec_hz,
    #                                         frequency_stx_hz=frequency_inferno_hz,
    #                                         band_order_Nth=order_Nth,
    #                                         dictionary_type="spect")
    #
    # stx_power = 2*np.abs(stx_complex)**2
    # stx_log2_power = np.log2(stx_power + scales.EPSILON)
    # stx_log2_power -= np.max(stx_log2_power)
    #
    # pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
    #                        wf_panel_a_sig=acc_sig,
    #                        wf_panel_a_time=time_cwt_s,
    #                        mesh_time=time_cwt_s,
    #                        mesh_frequency=frequency_cwt_hz,
    #                        mesh_panel_b_tfr=cwt_log2_power,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="AccZ CWT for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)
    #
    # pltq.plot_wf_mesh_vert(redvox_id=str(station_id),
    #                        wf_panel_a_sig=acc_sig,
    #                        wf_panel_a_time=time_s,
    #                        mesh_time=time_stx_s,
    #                        mesh_frequency=frequency_stx_hz,
    #                        mesh_panel_b_tfr=stx_log2_power,
    #                        mesh_panel_b_colormap_scaling="range",
    #                        wf_panel_a_units="Norm",
    #                        mesh_panel_b_cbar_units="bits",
    #                        start_time_epoch=0,
    #                        figure_title="AccZ STX for " + EVENT_NAME,
    #                        frequency_hz_ymin=fmin,
    #                        frequency_hz_ymax=fmax)
    #
    # plt.show()
