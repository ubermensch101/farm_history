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

    # query = "select ogc_fid, crop_cycle_22_23, may_2022_crop_presence, jun_2022_crop_presence, jul_2022_crop_presence, aug_2022_crop_presence, sep_2022_crop_presence, oct_2022_crop_presence, nov_2022_crop_presence, dec_2022_crop_presence, jan_2023_crop_presence, feb_2023_crop_presence, mar_2023_crop_presence, apr_2023_crop_presence from pilot.bid_559207 where crop_cycle_22_23 = 'kharif_and_rabi' limit 20"
    # with pgconn.cursor() as curs:
    #     curs.execute(query)
    #     rows = curs.fetchall()

    # ogc_fid = [item[0] for item in rows]
    # for i in range(20):
    #     print("ogc_fid:", ogc_fid[i], [f"{int(100*rows[i][j])}%" for j in range(2,14)])

    query = """
    select
        crop_cycle_22_23,
        avg(may_2022_crop_presence),
        avg(jun_2022_crop_presence),
        avg(jul_2022_crop_presence),
        avg(aug_2022_crop_presence),
        avg(sep_2022_crop_presence),
        avg(oct_2022_crop_presence),
        avg(nov_2022_crop_presence),
        avg(dec_2022_crop_presence),
        avg(jan_2023_crop_presence),
        avg(feb_2023_crop_presence),
        avg(mar_2023_crop_presence),
        avg(apr_2023_crop_presence)
    from
        pilot.bid_558923
    where
        crop_cycle_22_23 is not null
    group by
        crop_cycle_22_23
    """
    with pgconn.cursor() as curs:
        curs.execute(query)
        avg_rows = curs.fetchall()

    query = """
    select
        crop_cycle_22_23,
        stddev(may_2022_crop_presence),
        stddev(jun_2022_crop_presence),
        stddev(jul_2022_crop_presence),
        stddev(aug_2022_crop_presence),
        stddev(sep_2022_crop_presence),
        stddev(oct_2022_crop_presence),
        stddev(nov_2022_crop_presence),
        stddev(dec_2022_crop_presence),
        stddev(jan_2023_crop_presence),
        stddev(feb_2023_crop_presence),
        stddev(mar_2023_crop_presence),
        stddev(apr_2023_crop_presence)
    from
        pilot.bid_558923
    where
        crop_cycle_22_23 is not null
    group by
        crop_cycle_22_23
    """
    with pgconn.cursor() as curs:
        curs.execute(query)
        var_rows = curs.fetchall()

    for avg, var in zip(avg_rows, var_rows):
        print("average:", avg[0], [int(100*avg[j]) for j in range(1,13)])
        print("variance:", var[0], [int(100*var[j]) for j in range(1,13)])

