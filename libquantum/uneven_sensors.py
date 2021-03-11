import numpy as np
import obspy.signal.filter
import pandas as pd
from libquantum import scales, utils
import scipy.signal as signal
import matplotlib.pyplot as plt


# Height from barometric pressure in kPa
def height_asl_from_pressure_below10km(bar_waveform: np.ndarray) -> np.ndarray:
    """
    Simple model for troposphere
    :param bar_waveform: barometric pressure in kPa
    :return: height ASL in m
    """
    mg_rt = 0.00012  # Molar mass of air x gravity / (gas constant x standard temperature)
    elevation_m = -np.log(bar_waveform/scales.Slice.PREF_KPA)/mg_rt
    return elevation_m


def model_height_from_pressure_skyfall(pressure_kPa):
    """
    Returns empirical height in m from input pressure
    :param pressure_kPa: barometric pressure in kPa
    :return: height in m
    """
    pressure_ref_kPa = 101.325
    scaled_pressure = -np.log(pressure_kPa/pressure_ref_kPa)
    # Empirical model constructed from
    # c, stats = np.polynomial.polynomial.polyfit(poly_x, bounder_loc['Alt_m'], 8, full=True)
    c = [1.52981286e+02, 7.39552295e+03, 2.44663285e+03, -3.57402081e+03, 2.02653051e+03,
         -6.26581722e+02, 1.11758211e+02, -1.08674469e+01, 4.46784010e-01]
    elevation_m = np.polynomial.polynomial.polyval(scaled_pressure, c, tensor=False)
    return elevation_m


# RC filter response: mag first contribution to stack overflow as slipstream
# https://stackoverflow.com/questions/62448904/how-to-implement-continuous-time-high-low-pass-filter-in-python
def rc_low_pass(x_new, y_old, sample_rate_hz, frequency_cut_high_hz):
    sample_interval_s = 1/sample_rate_hz
    rc = 1/(2 * np.pi * frequency_cut_high_hz)
    alpha = sample_interval_s/(rc + sample_interval_s)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new


def rc_high_pass(x_new, x_old, y_old, sample_rate_hz, frequency_cut_low_hz):
    sample_interval_s = 1/sample_rate_hz
    rc = 1/(2 * np.pi * frequency_cut_low_hz)
    alpha = rc/(rc + sample_interval_s)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new


def rc_iterator_highlow(sensor_wf, sample_rate_hz,
                        frequency_cut_low_hz,
                        frequency_cut_high_hz):
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0
    y_prev_low = 0

    for x in sensor_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz,
                                   frequency_cut_low_hz)
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz,
                                 frequency_cut_high_hz)
        x_prev = x
        yield y_prev_high, y_prev_low


def rc_iterator_highpass(sensor_wf, sample_rate_hz, frequency_cut_low_hz):
    # Initialize. This can be improved to match wikipedia.
    x_prev = 0
    y_prev_high = 0

    for x in sensor_wf:
        y_prev_high = rc_high_pass(x, x_prev, y_prev_high, sample_rate_hz,
                                   frequency_cut_low_hz)
        x_prev = x
        yield y_prev_high


def rc_iterator_lowpass(sensor_wf, sample_rate_hz,
                        frequency_cut_high_hz):
    # Initialize. This can be improved to match wikipedia.
    y_prev_low = 0

    for x in sensor_wf:
        y_prev_low = rc_low_pass(x, y_prev_low, sample_rate_hz,
                                 frequency_cut_high_hz)
        yield y_prev_low


# "Traditional" solution, up to Nyquist
def bandpass_butter_uneven(sensor_wf, filter_order, frequency_cut_low_hz, sample_rate_hz):
    # Frequencies are scaled by Nyquist, with 1 = Nyquist
    # filter_order = 4,
    nyquist = 0.5 * sample_rate_hz
    edge_low = frequency_cut_low_hz / nyquist
    edge_high = 0.5
    [b, a] = signal.butter(N=filter_order, Wn=[edge_low, edge_high], btype='bandpass')
    sensor_bandpass = signal.filtfilt(b, a, np.copy(sensor_wf))
    return sensor_bandpass


def highpass_obspy(sensor_wf, frequency_low_Hz, sample_rate_Hz, filter_order=4):
    sensor_highpass = obspy.signal.filter.highpass(np.copy(sensor_wf),
                                                   frequency_low_Hz,
                                                   sample_rate_Hz, corners=filter_order,
                                                   zerophase=True)
    return sensor_highpass


def highpass_from_diff(sensor_waveform: np.ndarray,
                       sensor_epoch_s: np.ndarray,
                       sample_rate: int or float,
                       highpass_type: str = 'obspy4z',
                       frequency_filter_low: float = 1./scales.Slice.T100S) \
        -> (np.ndarray, np.ndarray):
    """
    Preprocess barometer data:
    - remove nans and DC offset by getting the differential pressure in kPa
    - apply highpass filter at 100 second periods
    - reconstruct Pressure in kPa from differential pressure: P(i) = dP(i) + P(i-1), P(0) = 0
    :param sensor_waveform:
    :param sensor_epoch_s:
    :param sample_rate:
    :param highpass_type: 'obspy4', 'butter', 'rc'
    :param frequency_filter_low: 100s default
    :return:
    """

    # Apply diff to remove DC offset; difference of nans is a nan
    # Replace nans with zeros, otherwise most things don't run
    sensor_waveform_diff = utils.demean_nan(np.diff(sensor_waveform))

    # Override default high pass at 100 seconds if signal is too short
    # May be able to zero pad ... with ringing. Or fold as needed.
    if sensor_epoch_s[-1] - sensor_epoch_s[0] < 2/frequency_filter_low:
        frequency_filter_low = 2/(sensor_epoch_s[-1] - sensor_epoch_s[0])
        print('Default 100s highpass override. New highpass period = ', 1/frequency_filter_low)

    # Taper test:
    # bar_waveform_diff_taper = bar_waveform_diff*utils.taper_tukey(bar_waveform_diff, 0.2)
    # Apply highpass filter
    # Watch out for edge ringing
    if highpass_type == "obspy4z":
        sensor_waveform_dp_filtered = \
            obspy.signal.filter.highpass(sensor_waveform_diff,
                                         frequency_filter_low,
                                         sample_rate, corners=4,
                                         zerophase=True)
    # bar_waveform_dp_filtered = bandpass_butter_uneven(bar_waveform_diff, filter_order=4,
    #                                                     frequency_low_Hz=frequency_filter_low,
    #                                                     sample_rate_Hz=sample_rate)

    # TODO: Construct Reconstruct Function
    # reconstruct from dP: P(0) = 0, P(i) = dP(i) + P(i-1)
    sensor_waveform_reconstruct = np.zeros((len(sensor_waveform)))

    # Initialize
    # sensor_waveform_reconstruct[0] = sensor_waveform_dp_filtered[0]

    # TODO: make sure this works if sensor_waveform_dp_filtered[i] is not set
    for i in range(1, len(sensor_waveform) - 1):
        sensor_waveform_reconstruct[i] = sensor_waveform_dp_filtered[i] + sensor_waveform_reconstruct[i-1]

    return sensor_waveform_reconstruct, frequency_filter_low


# TODO: Integrate with bar results
def sensor_uneven_3c(sensor_x, sensor_y, sensor_z, sensor_epoch, sensor_sample_rate):
    mag_waveform = [sensor_x, sensor_y, sensor_z]
    
    mag_epoch_diff = sensor_epoch[:-1]
    
    mag_diff = np.diff(mag_waveform)
    
    mag_waveform_dp_filtered_all = []
    mag_waveform_reconstruct_all = []
    
    for vector in range(len(mag_waveform)):
    
        # condition for when the record is too short for high pass at 100 seconds.
        # Needs to be at least 200 seconds
        if sensor_epoch[-1] - sensor_epoch[0] < 200:
            raise ValueError('Sensor record is too short for performing scalogram computations.' +
                             'Please provide a record longer than 200 seconds (3 minutes and 20 seconds).')
    
        # apply high pass filter at 100 seconds for sample rate calculated from the mag sensor.
        mag_waveform_dp_filtered = obspy.signal.filter.highpass(mag_diff[vector], 0.01, sensor_sample_rate, corners=4,
                                                                zerophase=True)
        mag_waveform_dp_filtered_all.append(mag_waveform_dp_filtered)
    
        # mag data reconstruct reconstruct mag from dP: P(0) = 0, P(i) = dP(i) + P(i-1)
        mag_waveform_reconstruct = np.zeros(len(mag_epoch_diff))
        mag_waveform_reconstruct[0] = 0
    
        # loop over the time series and reconstruct each data point
        for j in range(1, len(mag_waveform_reconstruct)):
            mag_waveform_reconstruct[j] = mag_waveform_dp_filtered[j] + mag_waveform_reconstruct[j - 1]
    
        mag_waveform_reconstruct_all.append(mag_waveform_reconstruct)
    
    # calculate and subtract the mean of each vector from each vector to remove the "DC offset".
    # Loop through each vector.
    mag_waveform_mean_removed_all = []
    
    for vector in range(len(mag_waveform_reconstruct_all)):
        vector_mean = np.mean(mag_waveform_reconstruct_all[vector])
    
        # subtract the mean of each vector from each value in that vector
        mag_waveform_mean_removed = []
        for j in range(len(mag_waveform_reconstruct_all[vector])):
            mag_waveform_mean_removed = mag_waveform_reconstruct_all[vector] - vector_mean
        mag_waveform_mean_removed_all.append(mag_waveform_mean_removed)
    
    # calculate/ display the mean of each vector AFTER removing the mean
    mag_x_mean_removed = mag_waveform_mean_removed_all[0]
    mag_y_mean_removed = mag_waveform_mean_removed_all[1]
    mag_z_mean_removed = mag_waveform_mean_removed_all[2]
    
    # calculate the max of all the components for use in the plot limits later on.
    mag_x_max_of_mean = np.max(mag_x_mean_removed)
    mag_y_max_of_mean = np.max(mag_y_mean_removed)
    mag_z_max_of_mean = np.max(mag_z_mean_removed)
    max_all_components = [mag_x_max_of_mean, mag_y_max_of_mean, mag_z_max_of_mean]
    max_final = np.round(np.max(max_all_components))
    min_final = -max_final

    # plot setup for displaying the mean removed signal from each component
    df = pd.DataFrame({'x': mag_epoch_diff, 'x-component': mag_x_mean_removed, 'y-component': mag_y_mean_removed,
                       'z-component': mag_z_mean_removed})

    # Calculate mag intensity from all three components
    mag_intensity_all = []
    for j in range(len(mag_x_mean_removed)):
        mag_intensity = np.sqrt(((mag_x_mean_removed[j]) ** 2) + ((mag_y_mean_removed[j]) ** 2) +
                                ((mag_z_mean_removed[j]) ** 2))
        mag_intensity_all.append(mag_intensity)
    
    # plot setup for mag intensity plot
    df2 = pd.DataFrame({'x': mag_epoch_diff, 'y': mag_intensity_all})

    # # calculate the fft
    # mag_sample_rate = [mag_sample_rate]
    # mag_intensity_array = [np.array(mag_intensity_all)]
    # time_fft, frequency_fft, energy_fft, snr_fft = spectra.spectra_fft(mag_intensity_array, mag_sample_rate,
    #                                                                    redvox_id, minimum_frequency=0.1)
    #
    # # calculate fft limits for plotting
    # max_energy = (np.amax(energy_fft))
    # min_energy = (np.amin(energy_fft))
