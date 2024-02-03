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


    tables = config.setup_details["tables"]["villages"]
    for table in tables:
        months_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        months_num = ['01', '02', '03', '04', '05', '06',
                      '07', '08', '09', '10', '11', '12']
        fetch_months = config.setup_details["months"]["agricultural_months"]
        column_check = False
        for month in fetch_months:
            column_check = column_check or \
                check_column_exists(pgconn_obj, table["schema"], table["table"],
                    months_name[month[1]] + '_' + str(month[0]) + '_red') or \
                check_column_exists(pgconn_obj, table["schema"], table["table"],
                    months_name[month[1]] + '_' + str(month[0]) + '_green') or \
                check_column_exists(pgconn_obj, table["schema"], table["table"],
                    months_name[month[1]] + '_' + str(month[0]) + '_blue') or \
                check_column_exists(pgconn_obj, table["schema"], table["table"],
                    months_name[month[1]] + '_' + str(month[0]) + '_nir')
            
        print("Processing table", f'{table["schema"]}.{table["table"]}')
        if column_check:
            print("Band columns exist. Replacing")
            for month in fetch_months:
                sql_query = f"""
                alter table {table["schema"]}.{table["table"]}
                drop column if exists {months_name[month[1]]}_{month[0]}_green;
                alter table {table["schema"]}.{table["table"]}
                drop column if exists {months_name[month[1]]}_{month[0]}_blue;
                alter table {table["schema"]}.{table["table"]}
                drop column if exists {months_name[month[1]]}_{month[0]}_red;
                alter table {table["schema"]}.{table["table"]}
                drop column if exists {months_name[month[1]]}_{month[0]}_nir;
                """
                with pgconn.cursor() as curs:
                    curs.execute(sql_query)
        else:
            print("Creating band columns")
        
        for month in fetch_months:
            sql_query = f"""
            alter table {table["schema"]}.{table["table"]}
            add column {months_name[month[1]]}_{month[0]}_red numeric;
            alter table {table["schema"]}.{table["table"]}
            add column {months_name[month[1]]}_{month[0]}_green numeric;
            alter table {table["schema"]}.{table["table"]}
            add column {months_name[month[1]]}_{month[0]}_blue numeric;
            alter table {table["schema"]}.{table["table"]}
            add column {months_name[month[1]]}_{month[0]}_nir numeric;
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
        
        sql_query = f"""
        select
            {table["key"]},
            st_astext(st_transform({table["geom_col"]}, 3857))
        from
            {table["schema"]}.{table["table"]}
        where
            description = 'field'
        and
            st_area({table["geom_col"]}::geography) > 1000
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            poly_fetch_all = curs.fetchall()

        print(len(poly_fetch_all))
        for month in fetch_months:
            print(month)
            for poly in poly_fetch_all:
                raster_path = f"data_bid/global_monthly_{month[0]}_{months_num[month[1]]}_mosaic/1453-1133_quad.tif"
                output_path = 'temp_clipped.tif'
                multipolygon = loads(poly[1])
                clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
                tif_path = output_path
                bands = calculate_average_color(tif_path)
                sql_query = f"""
                update
                    {table["schema"]}.{table["table"]}
                set
                    {months_name[month[1]]}_{month[0]}_red = {bands[0]},
                    {months_name[month[1]]}_{month[0]}_green = {bands[1]},
                    {months_name[month[1]]}_{month[0]}_blue = {bands[2]},
                    {months_name[month[1]]}_{month[0]}_nir = {bands[3]}
                where
                    {table["key"]} = {poly[0]}
                """
                with pgconn.cursor() as curs:
                    curs.execute(sql_query)
        pgconn.commit()