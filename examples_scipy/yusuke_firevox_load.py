import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input_files = ['20200222_FireVox_annotated_data.pkl',
    #                '20200210_FireVox_annotated_data.pkl',
    #                '20200711_FireVox_A_annotated_data.pkl',
    #                '20200711_FireVox_B_annotated_data.pkl',
    #                '20200711_FireVox_C_annotated_data.pkl',
    #                '20200711_FireVox_D_annotated_data.pkl',
    #                '20200711_FireVox_E_F_annotated_data.pkl']

    input_files = ['20200210_FireVox_annotated_data.pkl',
                   '20200711_FireVox_A_annotated_data.pkl',
                   '20200711_FireVox_B_annotated_data.pkl',
                   '20200711_FireVox_C_annotated_data.pkl',
                   '20200711_FireVox_D_annotated_data.pkl',
                   '20200711_FireVox_E_F_annotated_data.pkl']

    file_path_mg = "/Users/mgarces/Documents/DATA_API_M/FireVox_2020/RedVox_Milton/annotated_data/"
    file_path_m1 = "/Users/milton/Documents/DATA/FireVox_2020/RedVox_Milton/annotated_data/"

    for file_name in input_files:
        file_path = file_path_m1 + file_name
        df = pd.read_pickle(file_path)
        print("\nData file: ", file_path)
        print("Columns: ", df.columns)
        # print(pd.unique(df['class_id']))
        print("Metadata for annotated data")
        # print(f"Columns :{[i for i in df.columns]}")
        print(f"There are {df.shape[0]} annotated signals")
        unique_class = (pd.unique(df['class_id']))

        print("LEN SIG:", len(df['signal']))
        print(df.signal.dtypes)

        plt.plot(df['signal'][0][0])
        plt.show()
        #
        # for uclass in unique_class:
        #     sig_df = df[df['class_id'] == uclass]
        #     sig_df_id = sig_df.index
        #     sig_number_per_class = len(sig_df_id)
        #     print(uclass + ": " + str(sig_number_per_class))
        #     print(sig_df_id)
            # # plt.plot()
            # for j in sig_df_id:
            #     print("Index j:", j)
            #     plt.plot(np.asarray(df['signal'][j]))
            # # plt.show()

