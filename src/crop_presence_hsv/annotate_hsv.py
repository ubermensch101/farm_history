import os
import subprocess
import argparse
import cv2
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data")
BUFFERED_RASTER_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_hsv", "temp_buffered.tif")
RASTER_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_hsv", "temp_raw.tif")
HIGHLIGHT_PATH = os.path.join(ROOT_DIR, "src", "crop_presence_hsv", "temp_highlighted.tif")

# Create directory if doesn't exist
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
        os.makedirs(os.path.join(DATA_DIR,"train"))
        os.makedirs(os.path.join(DATA_DIR,"train", "n"))
        os.makedirs(os.path.join(DATA_DIR,"train", "y"))
        os.makedirs(os.path.join(DATA_DIR,"test"))
        os.makedirs(os.path.join(DATA_DIR,"test", "n"))
        os.makedirs(os.path.join(DATA_DIR,"test", "y"))
    except OSError as e:
        print(f"Error creating directory '{DATA_DIR}': {e}")

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()
    table = config.setup_details["tables"]["villages"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_key", type=int, required=True)
    parser.add_argument("-e", "--end_key", type=int, required=True)
    parser.add_argument("-d","--data_split", type=str, default="train")
    args = parser.parse_args()

    DATA_DIR = os.path.join(DATA_DIR, args.data_split)

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    annotate_months = config.setup_details["months"]["agricultural_months"]
    columns = [{table["key"]}]
    for month in annotate_months:
        columns.append(f"{months_names[month[1]]}_{month[0]}_crop_presence")

    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_multi(st_buffer(st_transform({table["geom_col"]}, 3857), 80))),
        st_astext(st_multi(st_transform({table["geom_col"]}, 3857)))
    from
        {table["schema"]}.{table["table"]}
    where
        {table["key"]} <= {args.end_key}
    and
        {table["key"]} >= {args.start_key}
    and
        {table["filter"]}
    order by
        {table["key"]}
    limit
        5
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    for farm in poly_fetch_all:

        for month in annotate_months:
            multipolygon = loads(farm[1])
            super_clip('quads', month[0], months_data[month[1]], multipolygon, BUFFERED_RASTER_PATH)
            raw_poly = farm[2]
            farmplot = loads(raw_poly)
            super_clip('quads', month[0], months_data[month[1]], farmplot, RASTER_PATH)
            polygon = [(float(item.split(' ')[0]), float(item.split(' ')[1])) for item in raw_poly.strip().split('(')[3].split(')')[0].split(',')]
            highlight_farm(BUFFERED_RASTER_PATH, polygon, HIGHLIGHT_PATH)

            file_name = "_".join([str(m) for m in month])
            file_name =  f"{farm[0]}_{file_name}.tif"
            while True:
                plt.ion()
                # cropped_path = crop_highlighted_farm(BUFFERED_RASTER_PATH, polygon)
                img = plt.imread(HIGHLIGHT_PATH)

                plt.figure(figsize=(10,6))
                plt.title(f'{table["key"]}: {farm[0]}, {months_names[month[1]]}-{month[0]}')
                plt.imshow(img)
                plt.draw()
                plt.pause(0.5)
                plt.close()

                answer = input()
                if answer in ["n", "y"]:
                    output_path = os.path.join(DATA_DIR, answer, file_name)
                    img = Image.open(RASTER_PATH)
                    img.save(output_path)
                    break  
                elif answer =="d":      ## In case of image containing clouds -> Drop it
                    break         
                else:
                    continue
    pgconn.commit()