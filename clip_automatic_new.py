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

import glob
import shapely
from shapely.geometry import shape

def get_available_rasters(directory):
    """
    Get a list of available raster files in the specified directory.

    Parameters:
        directory (str): Path to the directory containing raster files.

    Returns:
        List[str]: List of available raster file names.
    """
    raster_files = []
    for file in os.listdir(directory):
        if file.endswith('.tif'):  # Adjust file extension as needed
            raster_files.append(file)
    return raster_files

def raster_intersects_polygon(raster_path, polygon):
    """
    Check if a raster file intersects with a polygon.

    Parameters:
        raster_path (str): Path to the raster file.
        polygon (shapely.geometry.Polygon): Polygon geometry.

    Returns:
        bool: True if the raster intersects with the polygon, False otherwise.
    """
    with rasterio.open(raster_path) as src:
        # Get the raster bounding box
        raster_bbox = src.bounds

        # Create a shapely polygon from the raster bounding box
        raster_polygon = shape({
            "type": "Polygon",
            "coordinates": [
                [
                    [raster_bbox.left, raster_bbox.bottom],
                    [raster_bbox.right, raster_bbox.bottom],
                    [raster_bbox.right, raster_bbox.top],
                    [raster_bbox.left, raster_bbox.top],
                    [raster_bbox.left, raster_bbox.bottom]
                ]
            ]
        })

        # Check if the raster polygon intersects with the given polygon
        return raster_polygon.intersects(polygon)


if __name__ == '__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

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

    # Fetch all village polygons
    sql_query = f"""
    SELECT
        {table["key"]},
        ST_AsText(ST_Transform({table["geom_col"]}, 3857))
    FROM
        {table["schema"]}.{table["table"]}
    WHERE
        description = 'field'
    AND
        ST_Area({table["geom_col"]}::geography) > 1000
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    print(len(poly_fetch_all))
    for month in fetch_months:
        print(month)
        for poly in poly_fetch_all:
            multipolygon = loads(poly[1])
            for raster_name in get_available_rasters("/Users/adityatorne/Desktop/farm_history/farm_history/data_558923"):
                raster_path = f"{table['data_dir']}/global_monthly_{month[0]}_{months_num[month[1]]}_mosaic/{raster_name}"
                if raster_intersects_polygon(raster_path, multipolygon):
                    output_path = 'temp_clipped.tif'
                    clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
                    tif_path = output_path
                    bands = calculate_average_color(tif_path)

                    # Temporary bug fix. Further investigation required
                    # ogc_fid 584 in bid 559207
                    for i in range(len(bands)):
                        if bands[i] is None:
                            bands[i] = 0

                    sql_query = f"""
                    UPDATE
                        {table["schema"]}.{table["table"]}
                    SET
                        {months_name[month[1]]}_{month[0]}_red = {bands[0]},
                        {months_name[month[1]]}_{month[0]}_green = {bands[1]},
                        {months_name[month[1]]}_{month[0]}_blue = {bands[2]},
                        {months_name[month[1]]}_{month[0]}_nir = {bands[3]}
                    WHERE
                        {table["key"]} = {poly[0]}
                    """
                    with pgconn.cursor() as curs:
                        curs.execute(sql_query)
    pgconn.commit()

