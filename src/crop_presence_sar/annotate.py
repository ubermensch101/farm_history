import os
import subprocess
import argparse
import matplotlib.pyplot as plt
from shapely.wkt import loads
import tifffile as tiff
from config.config import Config
from utils.postgres_utils import PGConn
from utils.raster_utils import *

# FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
OUTPUT_DIR = os.path.join(ROOT_DIR, "sar_annotation")
DATA_DIR = os.path.join(ROOT_DIR, "sentinel_1")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Create directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' does not exist. Please run the sentinel-1.py script first.")

if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        os.makedirs(os.path.join(OUTPUT_DIR, "y"))
        os.makedirs(os.path.join(OUTPUT_DIR, "n"))
    except OSError as e:
        print(f"Error creating directory '{OUTPUT_DIR}': {e}")

if __name__ == '__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()
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
            {table["key"]} <= {args.end_key} and st_area(wkb_geometry::geography) > 10000
        and
            {table["key"]} >= {args.start_key}
        and
            {table["filter"]}
        order by
            {table["key"]}
        limit
            100
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
    
    RGB_QUADS_DIR = os.path.join(ROOT_DIR, args.interval, table['table'])
    SAR_QUADS_DIR = os.path.join(DATA_DIR, args.interval, table['table'])
    print("Quads dir:", SAR_QUADS_DIR)

    if not os.path.exists(os.path.join(CURRENT_DIR, "temp")):
        os.makedirs(os.path.join(CURRENT_DIR, "temp"))

    for farm in poly_fetch_all:
        key = farm[0]
        sar_images = []
        rgb_images = []
        for i in range(interval_length):
            sar_output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped_sar.tif'
            rgb_output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped_rgb.tif'
            multipolygon = loads(farm[1])
            super_clip_interval(SAR_QUADS_DIR, year, i+1, multipolygon, sar_output_path)
            super_clip_interval(RGB_QUADS_DIR, year, i+1, multipolygon, rgb_output_path)           
            try:
                sar_image = tiff.imread(sar_output_path)
                rgb_image = tiff.imread(rgb_output_path)
                sar_images.append(sar_image)
                rgb_images.append(rgb_image)
            except Exception as e:
                print(f"Error loading image at path {sar_output_path} or {rgb_output_path}: {e}")

        # Display images and process user input
        i = 0
        while i < interval_length:
            print(f"{args.interval[:-2]}:", i+1)
            num_rows = interval_length // 4
            fig, axes = plt.subplots(nrows=num_rows, ncols=8, figsize=(20, num_rows * 4))
            axes = axes.flatten()
            
            for j in range(interval_length):
                if j < len(sar_images):
                    sar_ax = axes[2 * j]
                    rgb_ax = axes[2 * j + 1]
                    sar_image = sar_images[j]  # Using SAR bands directly
                    rgb_image = rgb_images[j][:, :, :3]  # Using RGB bands directly
                    
                    sar_ax.imshow(sar_image)  # Transpose to match (height, width, bands)
                    rgb_ax.imshow(rgb_image)  # Display RGB image
                    
                    sar_ax.set_title(f'SAR {args.interval[:-2]}: {j+1}')
                    rgb_ax.set_title(f'RGB {args.interval[:-2]}: {j+1}')
                    
                    if i == j:
                        sar_ax.set_title(f'SAR {args.interval[:-2]}: {j+1}', color='r')
                        rgb_ax.set_title(f'RGB {args.interval[:-2]}: {j+1}', color='r')
                    
                    sar_ax.axis('off')
                    rgb_ax.axis('off')

            plt.suptitle(f"{table['key']}: {key}", fontsize=20)
            plt.tight_layout()
            plt.show()

            answer = input("Enter 'y' to save, 'n' to skip, 'd' to skip farm plot: ")
            if answer.lower() in ["y", "n"] and i < len(sar_images):
                file_name = f"{key}_{i+1}.tif"
                output_path = os.path.join(OUTPUT_DIR, answer.lower(), file_name)
                try:
                    tiff.imwrite(output_path, np.transpose(sar_images[i], (0, 1, 2)))
                    print(f"Saving to {output_path}")
                    i += 1
                except Exception as e:
                    print(f"Error saving image {i+1} to {output_path}: {e}")
            elif answer == "d":
                print("Skipping current farm plot...")
                break
            else:
                continue

            pgconn.commit()

    pgconn.commit()
