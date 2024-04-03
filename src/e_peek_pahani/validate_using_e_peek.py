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
        st_area(farm_geom)/10000,
        count(*)
    from
        nearest_farmplots_map
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

    confusion_matrix = np.array([np.zeros(3) for _ in range(3)])
    accuracy = {
        'toor': {
            'correct': 0,
            'wrong': 0
        },
        'harbhara': {
            'correct': 0,
            'wrong': 0
        },
        'soybean': {
            'correct': 0,
            'wrong': 0
        }
    }

    for row in rows:
        predicted_label = row[0]
        farmer = row[1:4]
        key = row[4]
        polygon = loads(row[5])
        area = row[6]
        sql_query = f"""
        select
            crop_name
        from
            nearest_farmplots_map
        join
            filtered_amravati_pahani
        using (
            khate_number,
            full_name,
            total_holding_area
        )
        where
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
                accuracy['toor']['correct'] += 1
            elif predicted_label == 'short_kharif':
                confusion_matrix[0][1] += 1
                accuracy['toor']['wrong'] += 1
            elif predicted_label == 'kharif_and_rabi':
                confusion_matrix[0][2] += 1

        if 'सोयाबीन' in crops and 'तुर' not in crops and 'कापुस' not in crops and 'हरभरा' not in crops:
            if predicted_label == 'short_kharif':
                confusion_matrix[1][1] += 1
                accuracy['soybean']['correct'] += 1
            elif predicted_label == 'long_kharif':
                confusion_matrix[1][0] += 1
                accuracy['soybean']['wrong'] += 1
            elif predicted_label == 'kharif_and_rabi':
                confusion_matrix[1][2] += 1
        
        if 'हरभरा' in crops:
            if predicted_label == 'kharif_and_rabi':
                confusion_matrix[2][2] += 1
                accuracy['harbhara']['correct'] += 1
            elif predicted_label == 'long_kharif':
                confusion_matrix[2][0] += 1
                accuracy['harbhara']['wrong'] += 1
            elif predicted_label == 'short_kharif':
                confusion_matrix[2][1] += 1

    print('toor case accuracy:')
    print(float(accuracy['toor']['correct'])/float(accuracy['toor']['correct'] + accuracy['toor']['wrong']))

    print('harbhara case accuracy')
    print(float(accuracy['harbhara']['correct'])/float(accuracy['harbhara']['correct'] + accuracy['harbhara']['wrong']))

    print('soybean case accuracy')
    print(float(accuracy['soybean']['correct'])/float(accuracy['soybean']['correct'] + accuracy['soybean']['wrong']))

    print('confusion matrix:')
    print(confusion_matrix)
