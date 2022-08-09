"""
Profile TFRs
# TODO: MAIN PARAMETERS FOR MAX CANVAS
# SAMPLE RATE -> max frequency, Nyquist
# AVERAGING FREQUENCY/PERIOD -> min frequency
# ORDER -> Sets duration of largest scale and hop interval/computation speed
# PIXELS PER MESH -> display graphics specification, sets resampling rate

"""
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from redvox.common.data_window import DataWindow
import redvox.common.date_time_utils as dtu
from libquantum import styx_stx, styx_cwt, scales
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq

# PROFILE SAMPLE RATE
profile_sample_rate = 48000.
EVENT_NAME = "Ukraine HIMARS, " + str(int(profile_sample_rate)) + "hz"
title_header = "HIMARS"
# Input Directories
input_dirs = ["/Users/mgarces/Documents/DATA_2022/ROCKET_ML/Ukraine_ML/dw_all_sample_rates.pkl.lz4"]
export_dir = "/Users/mgarces/Documents/DATA_2022/ROCKET_ML/Ukraine_ML/"
plot_audio_wf: bool = False
plot_tfr_mesh: bool = True

# Estimate amplitude in Pa
estimated_peak_db = 120
estimated_peak_pa = 20E-6 * 10**(estimated_peak_db/20)
print('Estimated peak pressure in Pa at full range (unity):', estimated_peak_pa)

# Gabor atom averaging spec (lowest frequency)
input_order = 6.
frequency_averaging_hz = 100.
number_cycles_averaging = 3*np.pi/4 * input_order
duration_averaging_s = number_cycles_averaging / frequency_averaging_hz

# Targeted Duration
duration_s = 60

# Bandwidth spec - build override
# frequency_cutoff_low_hz = frequency_averaging_hz
frequency_cutoff_high_hz = profile_sample_rate/2.

# Display spec
pixels_per_mesh = 2**16  # Edge spec is up to 2^20 for a single mesh target 2^19 pixels


def resampled_power_per_band_no_hop(sig_wf: np.array,
                                    sig_time: np.array,
                                    power_tfr: np.array,
                                    resampling_factor: int):

    """
    Assume len(sig_time) is a power of two
    Assume len(resampling_factor) is a power of two
    :param sig_wf: audio waveform
    :param sig_time: audio timestamps in seconds, same dimensions as sig
    :param power_tfr: time-frequency representation with same number of columns as sig
    :param resampling_factor: downsampling factor

    :return: rms_sig_wf, rms_sig_time_s
    """

    var_sig = (sig_wf - np.mean(sig_wf))**2

    if resampling_factor <= 1:
        print('No Resampling/Averaging!')
        exit()
    # Variance over the signal waveform, all bands
    # Number of rows (frequency)
    number_bands = power_tfr.shape[0]
    points_resampled: int = int(np.round((len(sig_wf)/resampling_factor)))
    var_wf_resampled = np.reshape(a=var_sig, newshape=(points_resampled, resampling_factor))

    var_sig_wf_mean = np.mean(var_wf_resampled, axis=1)
    var_sig_wf_max = np.max(var_wf_resampled, axis=1)
    var_sig_wf_min = np.min(var_wf_resampled, axis=1)

    # Reshape TFR
    var_tfr_resampled = np.reshape(a=power_tfr, newshape=(number_bands, points_resampled, resampling_factor))
    print('var_tfr_resampled.shape')
    print(var_tfr_resampled.shape)
    var_tfr_mean = np.mean(var_tfr_resampled, axis=2)
    var_tfr_max = np.max(var_tfr_resampled, axis=2)
    var_tfr_min = np.min(var_tfr_resampled, axis=2)

    # Resampled signal time
    var_sig_time_s = sig_time[0::resampling_factor]

    return [var_sig_time_s, var_sig_wf_mean, var_sig_wf_max, var_sig_wf_min, var_tfr_mean, var_tfr_max, var_tfr_min]


def main():
    """
    Load data window and select channel
    """

    for j, input_dir in enumerate(input_dirs):
        # Load data window from report
        print('Importing ', input_dir)
        dw = DataWindow.deserialize(input_dir)
        sequence_start_utc = dtu.datetime_from_epoch_seconds_utc(int(dw.start_date()*1E-6))
        # Get station data
        for station in dw.stations():
            audio_sample_rate_hz = station.audio_sample_rate_nominal_hz()
            if audio_sample_rate_hz != profile_sample_rate:
                continue
            else:
                # The main parameters can be computed immediately
                station_id = station.id()
                mic_points = station.audio_sensor().num_samples()

                print('\nStation ID:', station_id)
                print('Sample rate:', audio_sample_rate_hz)
                print('Number of sensor data samples:', mic_points)

                # This is a key parameter and sets the stride and the number of base atoms per record
                # It also sets the low-frequency edge of the TFR canvass
                ave_atom_points: int = int(duration_averaging_s*audio_sample_rate_hz)
                ave_atom_pow2_points: int = 2**int(np.ceil(np.log2(ave_atom_points)))

                mic_display_points: int = int(np.ceil(duration_s*audio_sample_rate_hz))

                # The Nyquist and averaging frequency - with the order - set the atom scales
                if frequency_cutoff_high_hz > audio_sample_rate_hz/2:
                    mic_frequency_cutoff_high_hz = audio_sample_rate_hz/2
                else:
                    mic_frequency_cutoff_high_hz = np.copy(frequency_cutoff_high_hz)

                # Get the standardized frequencies
                # TODO: Rewrite scales with Nyquist as reference
                order_Nth, cycles_M, frequency_center_geometric, frequency_start, frequency_end = \
                    styx_cwt.scale_frequency_bands(scale_order_input=input_order,
                                                   frequency_sample_rate_input=audio_sample_rate_hz,
                                                   frequency_low_input=frequency_averaging_hz,
                                                   frequency_high_input=mic_frequency_cutoff_high_hz)
                # Flip to match STFT
                frequency_inferno_hz = np.flip(frequency_center_geometric)

                # Number of frequency bands
                frequency_points = len(frequency_inferno_hz)

                # Have all the information needed
                # The desired number of display times is known, and the contraction factor can be computed
                time_contraction_factor: float = mic_display_points/(pixels_per_mesh/frequency_points)
                if time_contraction_factor > 1:
                    time_contraction_factor_pow2: int = 2**int(np.ceil(np.log2(time_contraction_factor)))
                else:
                    time_contraction_factor_pow2: int = 1
                    print('No averaging/resampling is needed for display')

                # Get the next relevant powers of two
                mic_points_pow2: int = 2**int(np.ceil(np.log2(mic_points)))
                mic_points_pow2_display: int = 2**int(np.ceil(np.log2(mic_display_points)))
                number_of_atoms_in_display: int = int(np.round(mic_points_pow2_display/ave_atom_pow2_points))

                print('Min, Max frequency:', [frequency_inferno_hz[0], frequency_inferno_hz[-1]])
                print('Time contraction factor, float:', time_contraction_factor)
                print('Time contraction factor, pow2:', time_contraction_factor_pow2)
                print('Number of atoms, display_pow2/atom_pow2', number_of_atoms_in_display)

                # Have not even loaded the audio data yet!
                # Get audio data
                audio_samples_0 = station.audio_sensor().get_microphone_data()
                # Remove mean
                audio_samples = audio_samples_0 - np.mean(audio_samples_0)
                audio_time_micros = station.audio_sensor().data_timestamps() - \
                                    station.audio_sensor().first_data_timestamp()
                # Convert to physical units
                audio_time_s = audio_time_micros*1E-6  # from microseconds to seconds
                audio_pa = audio_samples*estimated_peak_pa

                # Check waveform
                if plot_audio_wf:
                    # Plot input audio data
                    plt.figure()
                    plt.plot(audio_time_s, audio_pa)
                    plt.title(title_header + f", RedVox ID {station.id()}" + f", {int(audio_sample_rate_hz)} Hz")
                    plt.xlabel(f"Seconds from {sequence_start_utc} UTC")
                    plt.ylabel("Mic")
                    plt.show()

                # Zero pad relative to the largest of points_pow2
                # TODO: TEST for only a single atom and no resampling
                if mic_points_pow2_display < ave_atom_pow2_points:
                    if ave_atom_pow2_points > mic_points_pow2:
                        mic_pad_points = ave_atom_pow2_points - mic_points
                        mic_pad_sig = np.pad(array=audio_pa, pad_width=(mic_pad_points, 0))
                    else:
                        mic_pad_points = mic_points_pow2 - mic_points
                        mic_pad_sig = np.pad(array=audio_pa, pad_width=(mic_pad_points, 0))
                else:
                    if mic_points_pow2_display > mic_points_pow2:
                        mic_pad_points = mic_points_pow2_display - mic_points
                        mic_pad_sig = np.pad(array=audio_pa, pad_width=(mic_pad_points, 0))
                    else:
                        # TODO: TEST If exactly power of two, unchanged.
                        mic_pad_points = mic_points_pow2 - mic_points
                        mic_pad_sig = np.pad(array=audio_pa, pad_width=(mic_pad_points, 0))

                if plot_audio_wf:
                    # Plot zero padded data
                    plt.figure()
                    plt.plot(mic_pad_sig)
                    plt.title(title_header + f", RedVox ID {station.id()}" + f", {int(audio_sample_rate_hz)} Hz")
                    plt.xlabel(f"Seconds from {sequence_start_utc} UTC")
                    plt.ylabel("Mic")
                    plt.show()

                # For TFR plots
                if audio_sample_rate_hz < 100:
                    bit_range = 16
                else:
                    bit_range = 24

                # condition_no_resampling => no resampling, single atom
                condition_no_resample = (number_of_atoms_in_display <= 1) and (time_contraction_factor_pow2 <= 1)
                # condition_resample_one_atom => resampling, single atom
                condition_resample_one_atom = (number_of_atoms_in_display <= 1) and (time_contraction_factor_pow2 > 1)
                # condition_resample_atoms => resampling over more than one atom; can use atom as stride/hop
                condition_resample_atoms = (number_of_atoms_in_display > 1) and (time_contraction_factor_pow2 > 1)

                if condition_no_resample:
                    frequency_stx_hz, time_stx_s, stx_complex = \
                        styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_pad_sig,
                                                            frequency_sample_rate_hz=audio_sample_rate_hz,
                                                            frequency_stx_hz=frequency_inferno_hz,
                                                            band_order_Nth=order_Nth)

                    stx_power = 2*np.abs(stx_complex)**2
                    stx_log2_power2 = np.log2(stx_power + scales.EPSILON)
                    # Scale to max
                    stx_log2_power2 -= np.max(stx_log2_power2)

                    if plot_tfr_mesh:
                        pltq.plot_wf_mesh_vert(redvox_id=station_id,
                                               wf_panel_a_sig=mic_pad_sig,
                                               wf_panel_a_time=time_stx_s,
                                               mesh_time=time_stx_s,
                                               mesh_frequency=frequency_stx_hz,
                                               mesh_panel_b_tfr=stx_log2_power2,
                                               mesh_panel_b_colormap_scaling="range",
                                               mesh_panel_b_color_range=bit_range,
                                               wf_panel_a_units='Power',
                                               mesh_panel_b_cbar_units="bits",
                                               start_time_epoch=0,
                                               figure_title="Resampled Mic STX for " + EVENT_NAME,
                                               frequency_hz_ymin=frequency_inferno_hz[-0],
                                               frequency_hz_ymax=frequency_inferno_hz[-1])
                        plt.show()

                elif condition_resample_one_atom:
                    frequency_stx_hz, time_stx_s, stx_complex = \
                        styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_pad_sig,
                                                            frequency_sample_rate_hz=audio_sample_rate_hz,
                                                            frequency_stx_hz=frequency_inferno_hz,
                                                            band_order_Nth=order_Nth)
                    stx_power = 2*np.abs(stx_complex)**2
                    stx_log2_power2 = np.log2(stx_power + scales.EPSILON)
                    # Scale to max
                    stx_log2_power2 -= np.max(stx_log2_power2)

                    [var_sig_time_s,
                     var_sig_wf_mean, var_sig_wf_max, var_sig_wf_min,
                     var_tfr_mean, var_tfr_max, var_tfr_min] = \
                        resampled_power_per_band_no_hop(sig_wf=mic_pad_sig,
                                                        sig_time=time_stx_s,
                                                        power_tfr=stx_power,
                                                        resampling_factor=time_contraction_factor_pow2)

                    var_sig_wf = var_sig_wf_max
                    var_tfr = var_tfr_max

                    print('var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape')
                    print(var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape)
                    print(len(mic_pad_sig)/time_contraction_factor_pow2)

                    stx_log2_power2 = np.log2(var_tfr + scales.EPSILON)
                    stx_log2_power2 -= np.max(stx_log2_power2)

                    if plot_tfr_mesh:
                        pltq.plot_wf_mesh_vert(redvox_id=station_id,
                                               wf_panel_a_sig=var_sig_wf,
                                               wf_panel_a_time=var_sig_time_s,
                                               mesh_time=var_sig_time_s,
                                               mesh_frequency=frequency_stx_hz,
                                               mesh_panel_b_tfr=stx_log2_power2,
                                               mesh_panel_b_colormap_scaling="range",
                                               mesh_panel_b_color_range=bit_range,
                                               wf_panel_a_units='Power',
                                               mesh_panel_b_cbar_units="bits",
                                               start_time_epoch=0,
                                               figure_title="Resampled Mic STX for " + EVENT_NAME,
                                               frequency_hz_ymin=frequency_inferno_hz[-0],
                                               frequency_hz_ymax=frequency_inferno_hz[-1])
                        plt.show()

                elif condition_resample_atoms:
                    print("\nResample with ave atom stride")
                    # TODO: Ensure the resampled atom returns a reasonable value - may need another if
                    apply_atom_stride = True
                    if apply_atom_stride:
                        points_resampled: int = int(np.round((len(mic_pad_sig)/number_of_atoms_in_display)))
                        # mic_pad_per_atom = np.reshape(a=mic_pad_sig,
                        #                               newshape=(points_resampled, number_of_atoms_in_display))
                        #
                        # print('mic_pad_per_atom.shape', mic_pad_per_atom.shape)
                        # # exit()
                        # TODO: START HERE

                        var_sig_wf = []
                        var_tfr = []
                        for j in range(number_of_atoms_in_display):
                            print(j)
                            mic_pad_per_atom = mic_pad_sig[j*points_resampled:(j+1)*points_resampled]
                            print('mic_pad_per_atom.shape', mic_pad_per_atom.shape)

                            # plt.figure()
                            # plt.plot(mic_pad_per_atom)
                            # plt.show()
                            # # TODO: Harder/sharper window to reduce leakage?
                            frequency_stx_hz, time_stx_s, stx_complex = \
                                styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_pad_per_atom,
                                                                    frequency_sample_rate_hz=audio_sample_rate_hz,
                                                                    frequency_stx_hz=frequency_inferno_hz,
                                                                    band_order_Nth=order_Nth)
                            stx_power = 2*np.abs(stx_complex)**2

                            [var_sig_time_s,
                             var_sig_wf_mean, var_sig_wf_max, var_sig_wf_min,
                             var_tfr_mean, var_tfr_max, var_tfr_min] = \
                                    resampled_power_per_band_no_hop(sig_wf=mic_pad_per_atom,
                                                                    sig_time=time_stx_s,
                                                                    power_tfr=stx_power,
                                                                    resampling_factor=time_contraction_factor_pow2)

                            var_sig_wf.append(var_sig_wf_max)
                            var_tfr.append(var_tfr_max)

                            # print('var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape')
                            # print(var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape)
                            # print(len(mic_pad_sig)/time_contraction_factor_pow2)

                        # Add power, only compute at the end
                        var_tfr = np.concatenate(np.array(var_tfr), axis=1)
                        var_sig_wf = np.concatenate(np.array(var_sig_wf), axis=0)
                        print('final var_tfr.shape:', var_tfr.shape)
                        print('final var_sig_wf.shape:', var_sig_wf.shape)
                        # exit()

                        stx_log2_power2 = np.log2(np.array(var_tfr) + scales.EPSILON)
                        stx_log2_power2 -= np.max(stx_log2_power2)

                        var_sig_time_s = np.arange(stx_log2_power2.shape[1])/(audio_sample_rate_hz/time_contraction_factor_pow2)

                    else:
                        frequency_stx_hz, time_stx_s, stx_complex = \
                            styx_stx.stx_complex_any_scale_pow2(sig_wf=mic_pad_sig,
                                                                frequency_sample_rate_hz=audio_sample_rate_hz,
                                                                frequency_stx_hz=frequency_inferno_hz,
                                                                band_order_Nth=order_Nth)
                        stx_power = 2*np.abs(stx_complex)**2
                        stx_log2_power2 = np.log2(stx_power + scales.EPSILON)
                        # Scale to max
                        stx_log2_power2 -= np.max(stx_log2_power2)

                        [var_sig_time_s,
                         var_sig_wf_mean, var_sig_wf_max, var_sig_wf_min,
                         var_tfr_mean, var_tfr_max, var_tfr_min] = \
                            resampled_power_per_band_no_hop(sig_wf=mic_pad_sig,
                                                            sig_time=time_stx_s,
                                                            power_tfr=stx_power,
                                                            resampling_factor=time_contraction_factor_pow2)

                        var_sig_wf = var_sig_wf_max
                        var_tfr = var_tfr_max

                        print('var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape')
                        print(var_sig_time_s.shape, var_sig_wf.shape, var_tfr.shape)
                        print(len(mic_pad_sig)/time_contraction_factor_pow2)

                        stx_log2_power2 = np.log2(var_tfr + scales.EPSILON)
                        stx_log2_power2 -= np.max(stx_log2_power2)


                    if plot_tfr_mesh:
                        pltq.plot_wf_mesh_vert(redvox_id=station_id,
                                               wf_panel_a_sig=var_sig_wf,
                                               wf_panel_a_time=var_sig_time_s,
                                               mesh_time=var_sig_time_s,
                                               mesh_frequency=frequency_stx_hz,
                                               mesh_panel_b_tfr=stx_log2_power2,
                                               mesh_panel_b_colormap_scaling="range",
                                               mesh_panel_b_color_range=bit_range,
                                               wf_panel_a_units='Power',
                                               mesh_panel_b_cbar_units="bits",
                                               start_time_epoch=0,
                                               figure_title="Resampled Mic STX for " + EVENT_NAME,
                                               frequency_hz_ymin=frequency_inferno_hz[-0],
                                               frequency_hz_ymax=frequency_inferno_hz[-1])
                        plt.show()


if __name__ == "__main__":
    main()
