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
import pandas as pd

from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    
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
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly = curs.fetchall()[0]
        key = poly[0]

    images = []
    for i in range(24):
        output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped.tif'
        multipolygon = loads(poly[1])
        super_clip('fortnightly', None, i+1, multipolygon, output_path)
        images.append(np.array(Image.open(output_path)))
    
    columns = ""
    for i in range(24):
        columns += f"crop_presence_{i+1},"

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
    
    fig, axes = plt.subplots(nrows=6, ncols=4, figsize=(12,8))
    axes = axes.flatten()

    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image*2)
        ax.set_title(f'fortnight: {i+1}, prob: {round(record[i], 3)}')
    
    plt.suptitle(f"{table['key']}: {key}; cropping pattern: {record[-1]}", fontsize=20)
    plt.tight_layout()
    plt.show()