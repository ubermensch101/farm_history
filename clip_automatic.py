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
    
    def __del__(self):
        """No need to explicitly close the connection.  It will be closed when
        the PGConn object is garbage collected by Python runtime.

        """
        print(self.conn)
        self.conn.close()
        self.conn = None

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

def calculate_average_color(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()

        if data.shape[0] >= 4:
            red_avg = np.mean(data[0])
            green_avg = np.mean(data[1])
            blue_avg = np.mean(data[2])
            nir_avg = np.mean(data[3])
            return (red_avg, green_avg, blue_avg, nir_avg)
        else:
            raise ValueError("Need 4 bands corresponding to rgb and near-IR")

if __name__=='__main__':
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
        st_astext(st_transform(geom, 3857)) as geom_text
    from
        pilot.dagdagad_farmplots_dedup
    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    months_name = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    months_num = ['01', '02', '03', '04', '05', '06',
                  '07', '08', '09', '10', '11', '12']

    fetch_months = []
    months = [int(item) for item in np.linspace(0,11,num=12)]
    for i in months:
        fetch_months.append((2022,i))
        fetch_months.append((2023,i))

    results = []
    for month in fetch_months:
        # New table created with the average bands of each farmplot
        table = f'pilot.average_bands_dagdagad'
        print(month)
        print(len(results))
        for gid_poly in poly_fetch_all:
            raster_path = f'data/global_monthly_{month[0]}_{months_num[month[1]]}_mosaic/1465-1146_quad.tif'
            output_path = 'temp_clipped.tif'
            multipolygon = loads(gid_poly[1])
            clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
            tif_path = output_path
            rgb_nir = calculate_average_color(tif_path)
            results.append([gid_poly[0], months_name[month[1]], month[0],
                            rgb_nir[0], rgb_nir[1], rgb_nir[2], rgb_nir[3]])

    sql_query = f"""
    drop table if exists {table};
    create table {table} (
        gid integer,
        month text,
        year integer,
        red numeric,
        green numeric,
        blue numeric,
        nir numeric
    )
    """

    with pgconn.cursor() as curs:
        curs.execute(sql_query)

    for result in results:
        sql_query = f"""
        insert into {table} (
            gid,
            month,
            year,
            red,
            green,
            blue,
            nir
        )
        values (
            {result[0]},
            '{result[1]}',
            {result[2]},
            {result[3]},
            {result[4]},
            {result[5]},
            {result[6]}
        )
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)

    pgconn.commit()
