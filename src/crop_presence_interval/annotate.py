import os
import subprocess
import argparse
import cv2
import tifffile
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "fortnightly_data")
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
    parser.add_argument("-s", "--start_key", type=int, default=None)
    parser.add_argument("-e", "--end_key", type=int, default=None)
    parser.add_argument("-i", "--interval", type=str, default="monthly")
    args = parser.parse_args()

    year = config.setup_details['months']['agricultural_months'][0][0]

    args.interval = args.interval.lower()
    if args.interval == "fortnightly":
        interval_length = 24
    elif args.interval == "monthly":
        interval_length = 12
    elif args.interval == "weekly":
        interval_length = 52

    if args.start_key and args.end_key:
        sql_query = f"""
        select
            {table["key"]},
            st_astext(st_multi(st_transform({table["geom_col"]}, 4674)))
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
            2
        """
    else:
        sql_query = f"""
        select
            {table["key"]},
            st_astext(st_multi(st_transform({table["geom_col"]}, 4674)))
        from
            {table["schema"]}.{table["table"]}
        where
            {table["filter"]}
        order by
            random()
        limit
            2
        """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()
    
    QUADS_DIR = os.path.join(ROOT_DIR, args.interval, table['table'])
    print("Quads dir:", QUADS_DIR)

    for farm in poly_fetch_all:
        key = farm[0]
        images = []
        for i in range(interval_length):
            output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped.tif'
            multipolygon = loads(farm[1])
            super_clip_interval(QUADS_DIR,year , i+1, multipolygon, output_path)
            images.append(np.array(Image.open(output_path)))
        i = 0
        while(i < interval_length):
            print(f"{args.interval[:-2]}:", i+1)
            fig, axes = plt.subplots(nrows=interval_length//4, ncols=4, figsize=(12,8))
            axes = axes.flatten()
            for j, (ax, image) in enumerate(zip(axes, images)):
                ax.imshow(image*2)
                ax.set_title(f'fortnight: {j+1}')
            plt.suptitle(f"{table['key']}: {key}", fontsize=20)
            plt.tight_layout()
            plt.show()

            answer = input()
            if answer in ["n", "y"]:
                file_name = f"{key}_{i+1}.tif"
                output_path = os.path.join(DATA_DIR, answer, file_name)
                tifffile.imwrite(output_path, images[i])
                i += 1
            elif answer =="d":      ## In case of image containing clouds -> Drop it
                i += 1
            else:
                continue
    pgconn.commit()