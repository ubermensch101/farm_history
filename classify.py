import argparse
import csv
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

config = Config()
pgconn_obj = PGConn(config)
pgconn=pgconn_obj.connection()

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start_gid", type=int, required=True)
parser.add_argument("-e", "--end_gid", type=int, required=True)
args = parser.parse_args()

sql_query = f"""
select
    gid,
    st_astext(st_multi(st_transform(st_buffer(geom, 100), 3857))),
    st_astext(st_multi(st_transform(geom, 3857)))
from
    pilot.dagdagad_farmplots_dedup
where
    gid <= {args.end_gid}
and
    gid >= {args.start_gid}
"""

with pgconn.cursor() as curs:
    curs.execute(sql_query)
    poly_fetch_all = curs.fetchall()

df = pd.DataFrame(columns=['gid', 'month', 'year', 'crop_presence'])

months_data = ['01','02','03','04','05','06',
              '07','08','09','10','11','12']
months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
classify_months = config.setup_details["months"]["agricultural_months"]

for farm in poly_fetch_all:
    for month in classify_months:
        raster_path = f'data/global_monthly_{month[0]}_{months_data[month[1]]}_mosaic/1465-1146_quad.tif'
        multipolygon = loads(farm[1])
        output_path = 'temp_classify.tif'
        clip_raster_with_multipolygon(raster_path, multipolygon, output_path)
        tiff_file = output_path
        raw_poly = farm[2]
        polygon = [(float(item.split(' ')[0]), float(item.split(' ')[1])) for item in raw_poly.strip().split('(')[3].split(')')[0].split(',')]
        highlight_farm(tiff_file, polygon)

        while True:
            plt.ion()
            img = plt.imread('temp_classify.tif')
            plt.title(f'gid: {farm[0]}, month: {months_names[month[1]]}, {month[0]}')
            plt.imshow(img)
            plt.draw()
            plt.pause(1)
            plt.close()
            answer = input()
            if answer == 'y':
                df.loc[len(df)] = [farm[0], months_names[month[1]], month[0], 1]
                break
            if answer == 'n':
                df.loc[len(df)] = [farm[0], months_names[month[1]], month[0], 0]
                break
            if answer == 'r':
                continue

print("Writing results")
df.to_csv('classify_output.csv', index=False, mode='a', header=False)