import os
import csv

# Define the path to the main folder
main_folder_path = "/home/rahul/farm_history/data/crop_cycle_nanganur"

# Define the path to the CSV file
csv_file_path = "/home/rahul/farm_history/data/crop_data_nanganur.csv"

# Initialize a list to store folder names and subfolder names
folder_data = []

# Function to traverse through main folder and its subfolders
def traverse_main_folder(main_folder_path):
    # Get the list of folders inside the main folder
    main_folders = os.listdir(main_folder_path)
    
    # Iterate through each folder in the main folder
    for folder in main_folders:
        # Construct the full path of the folder
        folder_path = os.path.join(main_folder_path, folder)
        
        # Check if the item is a folder
        if os.path.isdir(folder_path):
            # Get the list of immediate subfolders
            subfolders = next(os.walk(folder_path))[1]
            
            # Iterate through each immediate subfolder
            for subfolder in subfolders:
                # Append the folder name and subfolder name
                folder_data.append([folder, subfolder])

# Start traversing from the main folder
traverse_main_folder(main_folder_path)

# Write the folder data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Main Folder Name', 'Subfolder Name'])
    writer.writerows(folder_data)

print("CSV file created successfully.")
# New column names
new_column_names = [
    "may_2022_crop_presence", "jun_2022_crop_presence", "jul_2022_crop_presence",
    "aug_2022_crop_presence", "sep_2022_crop_presence", "oct_2022_crop_presence",
    "nov_2022_crop_presence", "dec_2022_crop_presence", "jan_2023_crop_presence",
    "feb_2023_crop_presence", "mar_2023_crop_presence", "apr_2023_crop_presence"
]

# Read existing data from the CSV file
existing_data = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        existing_data.append(row)

# Add new column names to the header
header = existing_data[0]
header.extend(new_column_names)

# Write the updated data back to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Write the updated header
    writer.writerows(existing_data[1:])  # Write the remaining data

print("New columns added successfully.")
