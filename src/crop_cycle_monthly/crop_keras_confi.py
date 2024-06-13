import copy
import json
import math
import os
import pathlib
import time
import psycopg2
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry
import numpy as np
import rasterio
from shapely.wkt import loads
from rasterio.mask import mask
import cv2
import pickle
from tensorflow.keras.models import load_model

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

if __name__ == '__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()
    table = config.setup_details["tables"]["villages"][0]

    if not check_column_exists(pgconn_obj, table["schema"], table["table"], "crop_cycle_22_23"):
        sql_query = f"""
        ALTER TABLE
            {table["schema"]}.{table["table"]}
        ADD COLUMN
            crop_cycle_22_23 TEXT        
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    if not check_column_exists(pgconn_obj, table["schema"], table["table"], "confidence_22_23"):
        sql_query = f"""
        ALTER TABLE
            {table["schema"]}.{table["table"]}
        ADD COLUMN
            confidence_22_23 DOUBLE PRECISION        
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    if not check_column_exists(pgconn_obj, table["schema"], table["table"], "highest_prob"):
        sql_query = f"""
        ALTER TABLE
            {table["schema"]}.{table["table"]}
        ADD COLUMN
            highest_prob DOUBLE PRECISION        
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)        

    months_data = ['01', '02', '03', '04', '05', '06',
                   '07', '08', '09', '10', '11', '12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cycle_months = config.setup_details["months"]["agricultural_months"]
    columns = ""
    for month in cycle_months:
        columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"
        
    print(columns)
    
    # Load the Keras model
    model = load_model('/home/screa/farm_history/src/crop_cycle_monthly/weights/new_lstm_model.keras')
    
    crop_cycle_map = {
        0: "kharif_rabi",
        4: "short_kharif",
        1: "long_kharif",
        3: "perennial",
        6: "zaid",
        2: "no_crop",
        5: "weed"
    }

    sql_query = f"""
    SELECT
        {table["key"]}
    FROM
        {table["schema"]}.{table["table"]}
    WHERE
        {table["filter"]}
    ORDER BY
        {table["key"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        rows = curs.fetchall()
    keys = [item[0] for item in rows]
    print(keys)

    for key in keys:
        sql_query = f"""
        SELECT
            {columns}
            {table["key"]}
        FROM
            {table["schema"]}.{table["table"]}
        WHERE
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            row = curs.fetchall()[0]
        
        crop_presence_vec = np.array([[float(item) for item in row[0:-1]]])
        crop_presence_vec = np.expand_dims(crop_presence_vec, axis=1)
        
        # Make predictions and obtain confidence values
        predictions = model.predict(crop_presence_vec)
        print(predictions)
        sorted_indices = np.argsort(predictions)[0][-2:]  # Get indices of top two probabilities
        highest_prob_index = sorted_indices[-1]
        second_highest_prob_index = sorted_indices[-2]
        highest_prob = predictions[0][highest_prob_index]
        second_highest_prob = predictions[0][second_highest_prob_index]
        confidence_value = highest_prob / second_highest_prob

        crop_cycle = crop_cycle_map[highest_prob_index]

        sql_query = f"""
        UPDATE
            {table["schema"]}.{table["table"]}
        SET
            crop_cycle_22_23 = '{crop_cycle}',
            confidence_22_23 = {confidence_value},  -- Adding confidence value to the table
            highest_prob = {highest_prob}
        WHERE
            {table["key"]} = {key}
        """

        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    print('done')
