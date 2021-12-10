import pandas as pd
file_path = "/Users/mgarces/Documents/DATA_API_M/FireVox_2020/RedVox_Milton/annotated_data/20200711_Firevox_A_annotated_data.pkl"
df = pd.read_pickle(file_path)

print(pd.unique(df['class_id']))
print("Metadata for annotated data")
print(f"Columns :{[i for i in df.columns]}")
print(f"There are {df.shape[0]} annotated data")
print(f"There are {df[df['class_id'] == 'large'].shape[0]} data for large gunshot")
print(f"There are {df[df['class_id'] == 'medium'].shape[0]} data for medium gunshot")
print(f"There are {df[df['class_id'] == 'small'].shape[0]} data for small gunshot")