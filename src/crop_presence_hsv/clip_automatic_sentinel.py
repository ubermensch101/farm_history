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

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

def is_odd(num):
    return num % 2 != 0

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    months_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    fetch_months = config.setup_details["months"]["agricultural_months"]

    # Filter fetch_months to include only odd-numbered months
    fetch_months = [month for month in fetch_months if is_odd(month[1])]

    column_check = False
    for month in fetch_months:
        column_check = column_check or \
            check_column_exists(pgconn_obj, table["schema"], table["table"],
                                months_name[month[1]] + '_' + str(month[0]) + '_hue_mean') or \
            check_column_exists(pgconn_obj, table["schema"], table["table"],
                                months_name[month[1]] + '_' + str(month[0]) + '_hue_stddev')

    print("Processing table", f'{table["schema"]}.{table["table"]}')
    if column_check:
        print("Band columns exist. Replacing")
        for month in fetch_months:
            sql_query = f"""
            ALTER TABLE {table["schema"]}.{table["table"]}
            DROP COLUMN IF EXISTS {months_name[month[1]]}_{month[0]}_hue_mean;
            ALTER TABLE {table["schema"]}.{table["table"]}
            DROP COLUMN IF EXISTS {months_name[month[1]]}_{month[0]}_hue_stddev;
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    else:
        print("Creating band columns")

    for month in fetch_months:
        sql_query = f"""
        ALTER TABLE {table["schema"]}.{table["table"]}
        ADD COLUMN {months_name[month[1]]}_{month[0]}_hue_mean NUMERIC;
        ALTER TABLE {table["schema"]}.{table["table"]}
        ADD COLUMN {months_name[month[1]]}_{month[0]}_hue_stddev NUMERIC;
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    sql_query = f"""
    SELECT
        {table["key"]},
        ST_AsText(ST_Transform({table["geom_col"]}, 3857))
    FROM
        {table["schema"]}.{table["table"]}
    WHERE    
        {table["filter"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    print(len(poly_fetch_all))
    for month in fetch_months:
        print(month)
        count = 0
        for poly in poly_fetch_all:
            print(count)
            print(month)
            count += 1
            output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp_clipped.tif'
            multipolygon = loads(poly[1])
            # Use 'fortnightly' folder instead of 'quads'
            super_clip('fortnightly', month[0], months_name[month[1]], multipolygon, output_path)
            tif_path = output_path
            mean, stddev = compute_hue_features(tif_path)

            sql_query = f"""
            UPDATE
                {table["schema"]}.{table["table"]}
            SET
                {months_name[month[1]]}_{month[0]}_hue_mean = {mean},
                {months_name[month[1]]}_{month[0]}_hue_stddev = {stddev}
            WHERE
                {table["key"]} = {poly[0]}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()
