import os
import shutil
import subprocess
import argparse
import cv2
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *
import csv

if __name__ == "__main__":

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    months_data = [
        "01", "02", "03", "04", "05", "06",
        "07", "08", "09", "10", "11", "12",
    ]
    months_names = [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    study_months = config.setup_details["months"]["agricultural_months"]
    
    # Define the path to the existing CSV file
    csv_file_path = "/home/rahul/farm_history/data/crop_data_nanganur.csv"

    # Read the subfolder values from the CSV file
    subfolder_values = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subfolder_values.append(row['Subfolder Name'])
    
    # Convert subfolder_values to integers
    subfolder_values = list(map(int, subfolder_values))
    
    # List to store the fetched data
    data_to_append = []

    # Fetch data from PostgreSQL and append it to the list
    for k in subfolder_values:
        sql_query = f"""
        SELECT
            may_2022_crop_presence, jun_2022_crop_presence, jul_2022_crop_presence,
            aug_2022_crop_presence, sep_2022_crop_presence, oct_2022_crop_presence,
            nov_2022_crop_presence, dec_2022_crop_presence, jan_2023_crop_presence,
            feb_2023_crop_presence, mar_2023_crop_presence, apr_2023_crop_presence
        FROM
            {table["schema"]}.{table["table"]}
        WHERE
            id={k}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            poly_fetch_all = curs.fetchall()

        for poly in poly_fetch_all:
            # Append fetched data to the list preserving the first two columns
            data_to_append.append((k,) + poly)

    # Open the CSV file in read mode and read the existing data
    existing_data = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            existing_data.append(row)

    # Open the CSV file in write mode and write the existing data and new data
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in existing_data:
            writer.writerow(row)
        for row in data_to_append:
            writer.writerow(row)

    print("Data appended to CSV file successfully.")
