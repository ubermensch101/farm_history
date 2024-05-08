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

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    pgconn.commit()
    print("Processing table", f'{table["schema"]}.{table["table"]}')
    for i in range(24):
        sql_query = f"""
        alter table {table["schema"]}.{table["table"]}
        drop column if exists hue_mean_{i+1};
        alter table {table["schema"]}.{table["table"]}
        drop column if exists hue_stddev_{i+1};
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
    
    for i in range(24):
        sql_query = f"""
        alter table {table["schema"]}.{table["table"]}
        add column hue_mean_{i+1} numeric;
        alter table {table["schema"]}.{table["table"]}
        add column hue_stddev_{i+1} numeric;
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
    
    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_transform({table["geom_col"]}, 4674))
    from
        {table["schema"]}.{table["table"]}
    where    
        {table["filter"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()
    pgconn.commit()

    print(len(poly_fetch_all))
    for i in range(24):
        print("fortnight no:", i+1)
        for poly in poly_fetch_all:
            output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp_clipped.tif'
            multipolygon = loads(poly[1])
            super_clip('fortnightly', None, i+1, multipolygon, output_path)
            tif_path = output_path
            mean, stddev = compute_hue_features(tif_path)

            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                hue_mean_{i+1} = {mean},
                hue_stddev_{i+1} = {stddev}
            where
                {table["key"]} = {poly[0]}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()