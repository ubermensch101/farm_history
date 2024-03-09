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

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_key", type=int, required=True)
    parser.add_argument("-e", "--end_key", type=int, required=True)
    args = parser.parse_args()

    sql_query = f"""
    select
        {table["key"]}
    from
        {table["schema"]}.{table["table"]}
    where
        {table["key"]} >= {args.start_key}
    and
        {table["key"]} <= {args.end_key}
    and
        crop_cycle_22_23 is not null
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        keys = [item[0] for item in curs.fetchall()]

    df = pd.DataFrame(columns=[table["key"], "model_output", "actual_pattern"])

    crop_cycle_map = {
        'sk': "short_kharif",
        'lk': "long_kharif",
        'kr': "kharif_and_rabi",
        'p': "perennial"
    }

    for key in keys:
        sql_query = f"""
        select
            {table["key"]},
            st_astext(st_transform({table["geom_col"]}, 3857))
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {key}
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
            raster_path = f"{table['data_dir']}/global_monthly_{month[0]}_{months_data[month[1]]}_mosaic/{table['raster']}"
            output_path = f'temp/{months_names[month[1]]}_{month[0]}_clipped.tif'
            multipolygon = loads(poly[1])
            clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
            images.append(np.array(Image.open(output_path)))
            bands = calculate_average_color(output_path)
            nvdi.append((bands[3] - bands[0])/(bands[3] + bands[0]))
        
        columns = ""
        for month in study_months:
            columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"

        sql_query = f"""
        select
            {columns}
            crop_cycle_22_23
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            record = curs.fetchall()[0]

        actual_pattern = ""
        while True:
            fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12,8))
            axes = axes.flatten()

            for i, (ax, image, month) in enumerate(zip(axes, images, study_months)):
                ax.imshow(image)
                ax.set_title(f'{months_names[month[1]]}-{month[0]} prob: {round(record[i], 3)}')
            
            plt.suptitle(f"{table['key']}: {key}; cropping pattern: {record[-1]}", fontsize=20)
            plt.tight_layout()
            plt.show()
            answer = input()
            try:
                actual_pattern = crop_cycle_map[answer]
            except KeyError:
                print("Enter valid input")
                continue
            break

        df.loc[len(df)] = (key, record[-1], actual_pattern)

    df = df.to_csv('validation_report.csv', index=False, mode='a', header=False)