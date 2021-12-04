import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    # Set Path to pickled Data
    # export_path = '/Users/tokyok/Desktop/curated_explosions.pkl'
    export_path = '/Users/mgarces/Documents/DATA_API_M/Kei_blasts_20211202/curated_explosions.pkl'
    # export_path = '/PATH/TO/PICKLE//curated_explosions.pkl'

    # Import the pickled Data
    explosion_dataframe = pd.read_pickle(export_path)
    print(explosion_dataframe.columns.values)

    # Get unique event names
    unique_events = explosion_dataframe.event_name.unique()
    print(len(unique_events), ' events')
    print(len(explosion_dataframe.index.values), ' waveforms')

    # loop through each event to plot
    for event in unique_events:
        # select the data of said event
        event_dataframe = explosion_dataframe[explosion_dataframe.event_name == event]

        # Set up plot and loop through each station
        f, ax = plt.subplots(ncols=1, figsize=[12, 6], num=event)
        for num, id in enumerate(event_dataframe.index):
            # get time and normalized audio
            time = np.arange(len(event_dataframe['audio_raw'][id])) / event_dataframe['audio_sample_rate'][id]
            audio = event_dataframe['audio_raw'][id] / np.nanmax(event_dataframe['audio_raw'][id])

            # add to the plot and shift by num for wiggles
            ax.plot(time, audio + num * 2, 'black')

            # save title for first station
            if num == 0:
                title = event + ': effective yield = ' + str(event_dataframe.effective_yield_kg[id]) + ' kg'
                ax.set_title(title)

        # fancy plots
        ax.set_yticks(np.arange(len(event_dataframe.index)) * 2)
        ax.set_yticklabels(np.round(event_dataframe.scaled_distance_m.astype('float'), 2))
        ax.set_ylabel(r'scaled distance to source (m/kg$^\frac{1}{3}$)')
        ax.set_xlabel('relative time (s)')

    plt.show()


if __name__ == "__main__":
    main()
