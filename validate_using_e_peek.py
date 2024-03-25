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
        crop_cycle_22_23,
        khate_number,
        full_name,
        total_holding_area,
        key,
        st_astext(st_transform(farm_geom, 3857)),
        count(*)
    from
        nearest_kharif_farmplots_map
    join
        filtered_amravati_pahani
    using (
        khate_number,
        full_name,
        total_holding_area
    )
    group by
        khate_number,
        full_name,
        total_holding_area,
        key,
        farm_geom,
        crop_cycle_22_23
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        rows = curs.fetchall()

    confusion_matrix = np.array([
        [0,0],
        [0,0]
    ])

    total = len(rows)
    i = 0.0
    for row in rows:
        print(round(i*100/total, 3), '% completed')
        i += 1
        predicted_label = row[0]
        farmer = row[1:4]
        key = row[4]
        polygon = loads(row[5])
        sql_query = f"""
        select
            crop_name
        from
            nearest_kharif_farmplots_map
        join
            filtered_amravati_pahani
        using (
            khate_number,
            full_name,
            total_holding_area
        )
        where
            sowing_season_code = 2
        and
            khate_number = {farmer[0]}
        and
            full_name = '{farmer[1]}'
        and
            total_holding_area = {farmer[2]}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            crops = [item[0] for item in curs.fetchall()]

        if 'तुर' in crops:
            if predicted_label == 'long_kharif':
                confusion_matrix[0][0] += 1
            elif predicted_label == 'short_kharif':
                confusion_matrix[0][1] += 1

        if 'सोयाबीन' in crops and 'तुर' not in crops and 'कापुस' not in crops:
            if predicted_label == 'short_kharif':
                confusion_matrix[1][1] += 1
            elif predicted_label == 'long_kharif':
                confusion_matrix[1][0] += 1

    print('confusion matrix:')
    print(confusion_matrix)
