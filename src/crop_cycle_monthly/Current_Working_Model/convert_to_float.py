import pandas as pd

# Read the CSV file into a DataFrame, skipping the first column and first row
df = pd.read_csv("crop_data_train.csv", usecols=lambda column: column != 'ogc_fid', skiprows=1)

# Convert all entries (except the last column) to float values
for column in df.columns[:-1]:
    df[column] = df[column].astype(float)

# Specify the absolute path for the output CSV file
output_file = "/home/rahul/farm_history/data/crop_data_after_final.csv"

# Output the DataFrame to the new CSV file
df.to_csv(output_file, index=False)

print("Done")
