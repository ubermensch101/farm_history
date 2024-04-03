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

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    if not check_column_exists(pgconn_obj, table["schema"], table["table"], "crop_cycle_22_23"):
        sql_query = f"""
        alter table
            {table["schema"]}.{table["table"]}
        add column
            crop_cycle_22_23 text        
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cycle_months = config.setup_details["months"]["agricultural_months"]
    columns = ""
    for month in cycle_months:
        columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"

    model = load_model(f'{os.path.dirname(os.path.realpath(__file__))}/crop_cycle_predictor.keras')
    crop_cycle_map = {
        0: "short_kharif",
        1: "long_kharif",
        2: "kharif_and_rabi",
        3: "perennial"
    }

    sql_query = f"""
    select
        {table["key"]}
    from
        {table["schema"]}.{table["table"]}
    where
        {table["filter"]}
    order by
        {table["key"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        rows = curs.fetchall()
    keys = [item[0] for item in rows]

    for key in keys:
        sql_query = f"""
        select
            {columns}
            {table["key"]}
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            row = curs.fetchall()[0]
        
        crop_presence_vec = np.array([[float(item) for item in row[0:-1]],])
        crop_cycle_bits = model.predict(crop_presence_vec)
        crop_cycle = crop_cycle_map[np.argmax(crop_cycle_bits)]
        
        sql_query = f"""
        update
            {table["schema"]}.{table["table"]}
        set
            crop_cycle_22_23 = '{crop_cycle}'
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)