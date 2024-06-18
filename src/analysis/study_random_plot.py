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
        st_astext(st_transform({table["geom_col"]}, 3857))
    from
        {table["schema"]}.{table["table"]}
    where
        {table['filter']} AND ogc_fid=619
    order by
        random()
    limit
        1
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly = curs.fetchall()[0]
        key = poly[0]

    months_data = ['01','02','03','04','05','06',
                '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    study_months = config.setup_details["months"]["agricultural_months"]

    nvdi = []
    images = []
    for month in study_months:
        output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{months_names[month[1]]}_{month[0]}_clipped.tif'
        multipolygon = loads(poly[1])
        super_clip('quads', month[0], months_data[month[1]], multipolygon, output_path)
        images.append(np.array(Image.open(output_path)))
        bands = calculate_average_color(output_path)
        # Will maybe have to change this to average of nvdi instead of nvdi
        # of red and nir averages
        nvdi.append((bands[3] - bands[0])/(bands[3] + bands[0]))
    
    columns = ""
    for month in study_months:
        columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"
    
    columns+="crop_cycle_22_23"
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
        
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,8))
    axes = axes.flatten()

    for i, (ax, image, month) in enumerate(zip(axes, images, study_months)):
        ax.imshow(image)
        ax.set_title(f'{months_names[month[1]]}-{month[0]} prob: {round(record[i], 3)}')
    
    plt.suptitle(f"{table['key']}: {key}; cropping pattern: {record[-1]}", fontsize=20)
    plt.tight_layout()
    plt.show()