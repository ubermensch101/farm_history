import argparse
import copy
import json
import math
import os
import pathlib
import time
import psycopg2
import requests
import subprocess
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from shapely.wkt import loads
from rasterio.mask import mask
import cv2

from PIL import Image
import pandas as pd

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Clip and compute hue features for villages and add to database') 
    parser.add_argument('-f', '--fid', type=int, help='Farmplot ID', required=False,default=None)
    parser.add_argument("-i", "--interval", type=str, help="Interval type: fortnightly, monthly, weekly", default='monthly')  
    parser.add_argument('-y', '--year', type=int, help='Year', default=None)

    args = parser.parse_args()
    args.interval = args.interval.lower()


    if args.interval == "fortnightly":
        interval_length = 24
    elif args.interval == "monthly":
        interval_length = 12
    elif args.interval == "weekly":
        interval_length = 52
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    if args.year is None:  
        args.year = config.setup_details["months"]["agricultural_months"][0][0]


    if args.fid is None:
        sql_query = f"""
            select
                {table["key"]},
                st_astext(st_transform({table["geom_col"]}, 4674))
            from
                {table["schema"]}.{table["table"]}
            where
                {table['filter']}
            order by
                random()
            limit
                1
            """
    else:
        sql_query = f"""
            select
                {table["key"]},
                st_astext(st_transform({table["geom_col"]}, 4674))
            from
                {table["schema"]}.{table["table"]}
            where
                {table['key']} = {args.fid}
            """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly = curs.fetchall()[0]
        key = poly[0]

    images = []
    QUADS_DIR = os.path.join(ROOT_DIR, args.interval, table['table'])
    for i in range(interval_length):
        output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped.tif'
        multipolygon = loads(poly[1])
        super_clip_interval(QUADS_DIR, args.year, i+1, multipolygon, output_path)
        images.append(np.array(Image.open(output_path)))
    
    columns = ""
    for i in range(interval_length):
        columns += f"crop_presence_{args.year}_{args.interval}_{i+1},"

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
        record = curs.fetchall()[0]
    
    fig, axes = plt.subplots(nrows=interval_length//4, ncols=4, figsize=(12,8))
    axes = axes.flatten()

    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image)
        ax.set_title(f'[{args.interval[:2]}]: {i+1}, prob: {round(record[i], 3)}')
    
    plt.suptitle(f"{table['key']}: {key}; cropping pattern: {record[-1]}", fontsize=20)
    plt.tight_layout()
    plt.show()