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

    farmer_query = f"""
    select
        khate_number,
        full_name,
        total_holding_area,
        geom
    from
        filtered_amravati_pahani
    where
        sowing_season_code = 2
    group by
        khate_number,
        full_name,
        total_holding_area,
        geom
    """
    with pgconn.cursor() as curs:
        curs.execute(farmer_query)
        farmers = curs.fetchall()
    print(len(farmers), "farmers")

    mapping = []

    for farmer in farmers:
        point_geom = farmer[3]
        total_holding_area = farmer[2]
        farmplots_query = f"""
        select
            key,
            st_distance(st_geomfromwkb({point_geom}), geom)
        from
            s2.filtered_farmplots_amravati
        where
            st_intersects(
                st_buffer(st_geomfromwkb({point_geom}), 50),
                geom
            )
        and
            st_area(geom)
        between
            0.95*{total_holding_area}*10000
        and
            1.05*{total_holding_area}*10000
        """
        with pgconn.cursor() as curs:
            curs.execute(farmplots_query)
            farmplots = curs.fetchall()
        farmplots = farmplots.sort(key = lambda x : x[1])
        if farmplots is not None:
            print(farmplots)
            exit(0)
        else:
            print('bluh')