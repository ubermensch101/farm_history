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
from utils.postgres_utils import PGConn
from utils.raster_utils import *

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    sql_query = f"""
    select
        gid,
        st_astext(st_transform(geom, 3857))
    from
        pilot.dagdagad_farmplots_dedup
    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    months_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months_num = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12']

    fetch_months = config.setup_details["months"]["all_months"]

    results = []
    for month in fetch_months:
        # New table created with the average bands of each farmplot
        table = f'pilot.average_bands_dagdagad'
        print(month)
        print(len(results))
        for gid_poly in poly_fetch_all:
            raster_path = f'data/global_monthly_{month[0]}_{months_num[month[1]]}_mosaic/1465-1146_quad.tif'
            output_path = 'temp_clipped.tif'
            multipolygon = loads(gid_poly[1])
            clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
            tif_path = output_path
            rgb_nir = calculate_average_color(tif_path)
            results.append([gid_poly[0], months_name[month[1]], month[0],
                            rgb_nir[0], rgb_nir[1], rgb_nir[2], rgb_nir[3]])

    sql_query = f"""
    drop table if exists {table};
    create table {table} (
        gid integer,
        month text,
        year integer,
        red numeric,
        green numeric,
        blue numeric,
        nir numeric
    )
    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)

    for result in results:
        sql_query = f"""
        insert into {table} (
            gid,
            month,
            year,
            red,
            green,
            blue,
            nir
        )
        values (
            {result[0]},
            '{result[1]}',
            {result[2]},
            {result[3]},
            {result[4]},
            {result[5]},
            {result[6]}
        )
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    pgconn.commit()
