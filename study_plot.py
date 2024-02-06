import argparse
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
import matplotlib.pyplot as plt
import rasterio
from shapely.wkt import loads
from rasterio.mask import mask
import cv2

from PIL import Image

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--key", type=int, required=True,
                        help="Provide the unique key identifying the farmplot")
    args = parser.parse_args()

    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_transform({table["geom_col"]}, 3857))
    from
        {table["schema"]}.{table["table"]}
    where
        {table["key"]} = {args.key}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly = curs.fetchall()[0]

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    study_months = config.setup_details["months"]["agricultural_months"]

    nvdi = []
    images = []
    for month in study_months:
        raster_path = f"data_bid/global_monthly_{month[0]}_{months_data[month[1]]}_mosaic/1453-1133_quad.tif"
        output_path = f'temp/{months_names[month[1]]}_{month[0]}_clipped.tif'
        multipolygon = loads(poly[1])
        clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
        images.append(np.array(Image.open(output_path)))
        bands = calculate_average_color(output_path)
        # Will maybe have to change this to average of nvdi instead of nvdi
        # of red and nir averages
        nvdi.append((bands[3] - bands[0])/(bands[3] + bands[0]))
    
    sql_query = f"""
    select
        crop_cycle_22_23
    from
        {table["schema"]}.{table["table"]}
    where
        {table["key"]} = {args.key}    
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        crop_cycle = curs.fetchone()[0]
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,8))
    axes = axes.flatten()

    for ax, image, month in zip(axes, images, study_months):
        ax.imshow(image)
        ax.set_title(f'{months_names[month[1]]}-{month[0]}')
    
    plt.suptitle(f"{table['key']}: {args.key}; cropping pattern: {crop_cycle}", fontsize=20)
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # plt.plot(nvdi)
    # plt.show()