import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_name = "/Users/mgarces/Documents/DATA_API_M/INL_7g/curated_explosions.pkl"

if __name__ == "__main__":
    df = pd.read_pickle(file_name)
    print("Number of signals: ", len(df.index))
    print("Panda column names")
    print(df.columns)

    # All in one
    # f, ax = plt.subplots()
    # for num, ind in enumerate(imported_dataframe.index):
    #     ax.plot(imported_dataframe['audio_raw'][ind] + num * 2, 'black')
    # plt.show()

    for ind in df.index:
        yield_kg = df["effective_yield_kg"][ind]
        range_scaled = df["scaled_distance_m"][ind]
        sample_rate_hz = df["audio_sample_rate"][ind]
        title_str = "INL blasts, eq yield =" + str(int(1E3*yield_kg)) + " g, scaled range =" + str(int(range_scaled)) + " m"
        time_s = np.arange(len(df['audio_raw'][ind]))/sample_rate_hz
        time_scaled = time_s/(yield_kg**(1/3))

        plt.figure()
        plt.plot(time_scaled, df['audio_raw'][ind], 'k')
        plt.xlabel("Scaled time")
        plt.title(title_str)
        plt.grid(True)

    plt.show()

