"""
Annotion script for crop presence interval data
This script displays images of farm plots for each interval (week/biweek/month)in the agricultural season
All the images for a farm plot are displayed together and the user is asked to input 'y' or 'n' for each image
"""


import os
import h5py
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
from skimage import io
## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data", "sentinel_annotation")
CUURENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Create directory if doesn't exist
if not os.path.exists(DATA_DIR):
    try:
        os.makedirs(DATA_DIR)
        os.makedirs(os.path.join(DATA_DIR,"y"))
        os.makedirs(os.path.join(DATA_DIR,"n"))
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

    year = config.setup_details['months']['agricultural_years'][0]

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

    if not os.path.exists(os.path.join(CUURENT_DIR, "temp")):
        os.makedirs(os.path.join(CUURENT_DIR, "temp"))
    for farm in poly_fetch_all:
        key = farm[0]
        images = []
        for i in range(interval_length):
            output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped.tif'
            multipolygon = loads(farm[1])
            super_clip_interval(QUADS_DIR,year , i+1, multipolygon, output_path)
            with rasterio.open(output_path) as src:
                img =src.read()
                images.append(np.transpose(img, (1,2,0)))
        i = 0
        while(i < interval_length):
            print(f"{args.interval[:-2]}:", i+1)
            fig, axes = plt.subplots(nrows=interval_length//4, ncols=4, figsize=(12,8))
            axes = axes.flatten()
            for j, (ax, image) in enumerate(zip(axes, images)):
                rgb_image = image[:,:,:3]
                ax.imshow(rgb_image)
                if i == j:
                    ax.set_title(f'{args.interval[:-2]}: {j+1}', color='r')
                ax.set_title(f'{args.interval[:-2]}: {j+1}')

            plt.suptitle(f"{table['key']}: {key}", fontsize=20)
            plt.tight_layout()
            plt.show()

            answer = input()
            if answer in ["n", "y"]:
                file_name = f"{key}_{i+1}.tiff"
                output_path = os.path.join(DATA_DIR, answer, file_name)
                img_temp = np.transpose(images[i], (2, 0, 1 ))
                save_multidimension_raster(img_temp, output_path)
                print(f"Saving to {output_path}")
                i += 1
            elif answer =="d":      ## In case of image containing clouds -> Drop it
                i += 1
                print("Skipping...")
            else:
                continue
    pgconn.commit()