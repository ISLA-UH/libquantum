"""
Extract and display data
"""
import os
from typing import Tuple
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt

from redvox.common.data_window import DataWindow
import redvox.common.date_time_utils as dtu
from libquantum import styx_stx, styx_cwt, styx_fft, scales
import libquantum.plot_templates.plot_time_frequency_reps_black as pltq

# Input Directories
input_dirs = ["/Users/mgarces/Documents/DATA_2022/ROCKET_ML/Ukraine_ML/dw_all_sample_rates.pkl.lz4"]
sequence_texts = ["Ukraine HIMARS"]

export_dir = "/Users/mgarces/Documents/DATA_2022/ROCKET_ML/Ukraine_ML/"
export_wav: bool = False
plot_audio_wf: bool = False

estimated_peak_db = 120
estimated_peak_pa = 20E-6 * 10**(estimated_peak_db/20)
print('Estimated peak pressure in Pa at full range (unity):', estimated_peak_pa)

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
pixels_per_mesh = 2**16  # Edge spec is up to 2^20 for a single mesh target 2^19 pixels

# TODO: SUMMARY
# TODO: If size(TFR) < pixel per mesh, don't contract
# TODO: Make points_per_seg a power of two, and greater than or equal to 2, with checks

# TODO: Compute the max/min power per time step

def resampled_power_per_band(sig_wf: np.array,
                             sig_time: np.array,
                             power_tfr: np.array,
                             points_per_seg: int,
                             points_hop: int = None) -> Tuple[np.array, np.array, np.array]:
    """
    Look at
    https://localcoder.org/rolling-window-for-1d-arrays-in-numpy
    :param sig_wf: audio waveform
    :param sig_time: audio timestamps in seconds, same dimensions as sig
    :param power_tfr: time-frequency representation with same number of columns as sig
    :param points_per_seg: number of points, set by downsampling factor
    :param points_hop: number of points overlap per window. Default is 50%

    :return: rms_sig_wf, rms_sig_time_s
    """


    number_bands = power_tfr.shape[0]

    if points_hop is None:
        points_hop: int = points_per_seg  # Tested at 0.5*int(points_per_seg)

    var_sig = (sig_wf - np.mean(sig_wf))**2
    # https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html
    # Variance over the signal waveform, all bands
    var_wf_windowed = \
        np.lib.stride_tricks.sliding_window_view(var_sig,
                                                 window_shape=points_per_seg)[0::points_hop, :].copy()

    # TODO: Compute the max/min power per time step
    var_sig_wf = var_wf_windowed.mean(axis=-1)

    # Initialize
    var_tfr = np.zeros((number_bands, len(var_sig_wf)))
    # Mean of absolute TFR power per band
    for j in np.arange(number_bands):
        # TODO: Compute the max/min power per band per time step
        var_tfr_windowed = \
            np.lib.stride_tricks.sliding_window_view(power_tfr[j, :],
                                                     window_shape=points_per_seg)[0::points_hop, :].copy()
        var_tfr[j, :] = var_tfr_windowed.mean(axis=-1)

    # sig time
    var_sig_time_s = sig_time[0::points_hop].copy()

    # check dims
    diff = abs(len(var_sig_time_s) - len(var_sig_wf))

    if (diff % 2) != 0:
        var_sig_time_s = var_sig_time_s[0:-1]
    else:
        var_sig_time_s = var_sig_time_s[1:-1]

    return var_sig_time_s, var_sig_wf, var_tfr


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


def main():
    """
    Load audio
    """

    for j, input_dir in enumerate(input_dirs):
        # Load data window from report
        print('Importing ', input_dir)
        dw = DataWindow.deserialize(input_dir)
        sequence_start_utc = dtu.datetime_from_epoch_seconds_utc(int(dw.start_date()*1E-6))
        title_header = sequence_texts[j]

        # Get station data
        for station in dw.stations():
            # Get audio data
            audio_samples_0 = station.audio_sensor().get_microphone_data()
            # Remove mean
            audio_samples = audio_samples_0 - np.mean(audio_samples_0)
            audio_time_micros = station.audio_sensor().data_timestamps() - \
                                station.audio_sensor().first_data_timestamp()
            audio_time_s = audio_time_micros*1E-6  # from microseconds to seconds
            audio_sample_rate_hz = station.audio_sample_rate_nominal_hz()

            audio_pa = audio_samples*estimated_peak_pa

            # Check recording against input data
            if export_wav:
                if audio_sample_rate_hz > 16000:
                    wave_filename = 'ukraine_48khz.wav'
                    export_filename = os.path.join(export_dir + wave_filename)
                    synth_wav = 0.9 * np.real(audio_samples) / np.max(np.abs((np.real(audio_samples))))
                    scipy.io.wavfile.write(filename=export_filename, rate=48000, data=synth_wav)

            if plot_audio_wf:
                # Plot Audio data for each station
                plt.figure()
                plt.plot(audio_time_s, audio_pa)
                plt.title(title_header + f", RedVox ID {station.id()}" + f", {int(audio_sample_rate_hz)} Hz")
                plt.xlabel(f"Seconds from {sequence_start_utc} UTC")
                plt.ylabel("Mic")
                plt.show()

            # TODO: Zero Pad, process, truncate
            # TODO: Make this into a power of two
            # stx_time_contraction_factor = int(np.ceil((len(time_stx_s)/(pixels_per_mesh/len(frequency_inferno_hz)))))

if __name__ == "__main__":
    main()
