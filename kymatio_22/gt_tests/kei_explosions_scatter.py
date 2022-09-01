import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import libquantum.utils as utils
import libquantum.dyadics as dyadics
from kymatio.numpy import Scattering1D

# CONSTANTS
REF_HE_1KG_FREQUENCY_HZ: float = 50.
REF_HE_1KG_PERIOD_S: float = 0.02
REF_SAMPLE_RATE_HZ: float = 8000.0
REF_ORDER: int = 6
REF_T: int = 2**15
REF_J = 4

def main():
    # Set Path to pickled Data
    export_path = '/Users/milton/Documents/DATA/Kei_blasts_20211202/curated_explosions.pkl'

    # Import the pickled Data
    explosion_dataframe = pd.read_pickle(export_path)
    print(explosion_dataframe.columns.values)

    # Get unique event names
    unique_events = explosion_dataframe.event_name.unique()
    print(len(unique_events), ' events')
    print(len(explosion_dataframe.index.values), ' waveforms')

    # Initialize the Kymatio scattering tensor
    scattering = Scattering1D(REF_J, REF_T, REF_ORDER)

    meta = scattering.meta()
    order0 = np.where(meta['order'] == 0)
    order1 = np.where(meta['order'] == 1)
    order2 = np.where(meta['order'] == 2)

    # Use meta method to extract scaled frequencies
    xi = meta['xi']
    # Convert normalized frequency xi = f/fs to frequency
    freqs = xi*REF_SAMPLE_RATE_HZ

    # loop through each event to plot
    for event in unique_events:
        # select the data of said event
        event_dataframe = explosion_dataframe[explosion_dataframe.event_name == event]
        event_yield_kg = event_dataframe.effective_yield_kg[event_dataframe.index[0]]
        print("\nEvent " + event + " yield in kg: " + str(event_yield_kg))
        event_sach_scaling = event_yield_kg**(1/3)
        event_gt_peak_frequency_hz = REF_HE_1KG_FREQUENCY_HZ/event_sach_scaling
        event_gt_pseudoperiod_s = REF_HE_1KG_PERIOD_S*event_sach_scaling
        print("Predicted GT peak frequency, Hz: ", event_gt_peak_frequency_hz)
        print("Predicted GT pseudoperiod, s: ", event_gt_pseudoperiod_s)

        T_duration_s = \
            dyadics.duration_from_order_and_period(sig_period_s=event_gt_pseudoperiod_s,
                                                   order_number=REF_ORDER)
        T_points_log2, T_points_pow2, T_time_pow2_s = \
            dyadics.duration_floor(sample_rate_hz=REF_SAMPLE_RATE_HZ,
                                  time_s=T_duration_s)

        # Octave below
        J_points_log2, J_points_pow2, J_time_pow2_s = \
            dyadics.duration_floor(sample_rate_hz=REF_SAMPLE_RATE_HZ,
                                   time_s=event_gt_pseudoperiod_s/2)

        print("T duration, s: ", T_time_pow2_s)
        print("J duration, s: ", J_time_pow2_s)
        print("T pow2 number of points: ", T_points_pow2)
        print("Log2 T: ", T_points_log2)
        print("J: ", J_points_log2)


        # Set up plot and loop through each station
        # f, ax = plt.subplots(ncols=1, figsize=[12, 6], num=event)
        # time = []
        audio = []

        for num, event_id in enumerate(event_dataframe.index):
            sample_rate_raw_hz = event_dataframe['audio_sample_rate'][event_id]
            # print("\nSample rate, Hz: ", sample_rate_raw_hz)
            sample_rate_final_hz = 0.
            # get time and normalized audio, upsample 80/800 Hz to 8 kHz
            if sample_rate_raw_hz < REF_SAMPLE_RATE_HZ:
                resampled_aud = utils.upsample_fourier(sig_wf=event_dataframe['audio_raw'][event_id],
                                                       sig_sample_rate_hz=sample_rate_raw_hz,
                                                       new_sample_rate_hz=REF_SAMPLE_RATE_HZ)
                audio = resampled_aud / np.nanmax(resampled_aud)
                # time = np.arange(len(audio)) / REF_SAMPLE_RATE_HZ
                sample_rate_final_hz = REF_SAMPLE_RATE_HZ
                # print('Resampled rate, Hz: ', REF_SAMPLE_RATE_HZ)
            if sample_rate_raw_hz == REF_SAMPLE_RATE_HZ:
                # time = np.arange(len(event_dataframe['audio_raw'][event_id])) / event_dataframe['audio_sample_rate'][event_id]
                audio = event_dataframe['audio_raw'][event_id] / np.nanmax(event_dataframe['audio_raw'][event_id])
                sample_rate_final_hz = np.copy(sample_rate_raw_hz)
            if sample_rate_final_hz == 0.:
                print("Signal not processed: ", event_id)

            # CONDITION
            # Remove DC offset
            audio -= np.mean(audio)
            # Taper
            audio *= utils.taper_tukey(audio, fraction_cosine=0.1)
            # Center pad
            audio = librosa.util.pad_center(audio, size=REF_T, mode='constant')
            time = np.arange(len(audio)) / REF_SAMPLE_RATE_HZ
            # Convert to float32
            x = np.float32(audio)
            sample_rate = REF_SAMPLE_RATE_HZ
            # print(x.dtype, len(audio))

            # exit()
            # Scattering transform
            Sx = scattering(x)
            print("Scattering tensor shape:", Sx.shape)

            # This is what is used to
            # Split the solutions into zeroth, first, and second order
            Sx_0 = Sx[order0][0]
            Sx_1 = Sx[order1]
            Sx_2 = Sx[order2]
            frequency_0 = freqs[order0, 0]
            frequency_1 = np.squeeze(freqs[order1, 0])
            frequency_2 = freqs[order2]
            time_len = len(Sx_0)
            decimation_factor = len(x)/time_len

            time_x = np.arange(len(x))/sample_rate
            time_Sx = np.arange(len(Sx_0))/sample_rate*decimation_factor

            # # Print the zeroth, first, and second order solutions
            # print(frequency_0)
            # print(frequency_1)
            # print(frequency_2)
            # print(time_len, decimation_factor)

            # x = frequency_2[:, 0]
            # y = frequency_2[:, 1]
            # print(x)
            # print(y)

            # # Time series
            # plt.figure(figsize=(8, 2))
            # plt.plot(time_x, x)
            # plt.title("Original signal")

            # Scattering panels
            plt.figure(figsize=(8, 8))
            plt.subplot(3, 1, 1)
            plt.plot(Sx_0)
            plt.title('Zeroth-order scattering')
            plt.subplot(3, 1, 2)
            plt.imshow(Sx_1, aspect='auto')
            plt.title('First-order scattering')
            plt.subplot(3, 1, 3)
            plt.imshow(Sx_2, aspect='auto')
            plt.title('Second-order scattering')

            # Compare to first order to spectrogram
            plt.figure(figsize=(8, 8))
            # plt.subplot(1, 2, 1)
            plt.pcolormesh(time_Sx, frequency_1, Sx_1, shading='auto')
            plt.ylim(np.min(frequency_1), np.max(frequency_1))
            plt.title("First order")
            plt.yscale("log")
            # plt.subplot(1, 2, 2)
            # plt.specgram(x, Fs=sample_rate)
            # plt.ylim(np.min(frequency_1), np.max(frequency_1))
            # plt.title('Spectrogram')

            # Use scatter plot for second order?
            plt.show()

    #         # Stage plotting
    #         # add to the plot and shift by num for wiggles
    #         ax.plot(time, audio + num * 2, 'black')
    #
    #         # save title for first station
    #         if num == 0:
    #             title = event + ': effective yield = ' + str(event_dataframe.effective_yield_kg[event_id]) + ' kg'
    #             ax.set_title(title)
    #
    #     # fancy plots
    #     ax.set_yticks(np.arange(len(event_dataframe.index)) * 2)
    #     ax.set_yticklabels(np.round(event_dataframe.scaled_distance_m.astype('float'), 2))
    #     ax.set_ylabel(r'scaled distance to source (m/kg$^\frac{1}{3}$)')
    #     ax.set_xlabel('relative time (s)')
    #
    # plt.show()


if __name__ == "__main__":
    main()
