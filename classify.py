import copy
import csv
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

class PGConn:
    def __init__(self, host, port, dbname, user=None, passwd=None):
        self.host = host
        self.port = port
        self.dbname = dbname
        if user is not None:
            self.user = user
        else:
            self.user = ""
        if passwd is not None:
            self.passwd = passwd
        else:
            self.passwd = ""
        self.conn = None

    def connection(self):
        """Return connection to PostgreSQL.  It does not need to be closed
        explicitly.  See the destructor definition below.

        """
        if self.conn is None:
            conn = psycopg2.connect(dbname=self.dbname,
                                    host=self.host,
                                    port=str(self.port),
                                    user=self.user,
                                    password=self.passwd)
            self.conn = conn
            
        return self.conn

pgconn_obj = PGConn(
    "localhost",
    5432,
    "dolr",
    "sameer",
    "swimgood"
)
    
pgconn=pgconn_obj.connection()

sql_query = f"""
select
    gid,
    st_astext(st_transform(st_buffer(geom, 100), 3857)) as geom_text
from
    pilot.dagdagad_farmplots_dedup
where
    gid = 165
"""

with pgconn.cursor() as curs:
    curs.execute(sql_query)
    poly_fetch_all = curs.fetchall()

def clip_raster_with_multipolygon(raster_path, multipolygon, output_path):
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Clip the raster with the multipolygon
        out_image, out_transform = mask(src, [multipolygon], crop=True)
        
        # Copy the metadata
        out_meta = src.meta.copy()

        # Update the metadata to match the clipped raster
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})

        # Write the clipped raster to a new file
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

classifications = []

for farm in poly_fetch_all:
    months = ['01','02','03','04','05','06',
              '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for i,month in enumerate(months):
        raster_path = f'data/global_monthly_2023_{month}_mosaic/1465-1146_quad.tif'
        multipolygon = loads(farm[1])
        output_path = 'temp_classify.tif'
        clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
        while True:
            plt.ion()
            img = plt.imread('temp_classify.tif')
            plt.title(f'gid: {farm[0]}, month: {months_names[i]}')
            plt.imshow(img)
            plt.draw()
            plt.pause(1)
            plt.close()
            answer = input()
            if answer == 'y':
                classifications.append((farm[0], months_names[i], 1))
                break
            if answer == 'n':
                classifications.append((farm[0], months_names[i], 0))
                break
            if answer == 'r':
                continue

csv_file = 'classify_output.csv'
with open(csv_file, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(['gid', 'month', 'crop_presence'])
    for row in classifications:
        writer.writerow(row)