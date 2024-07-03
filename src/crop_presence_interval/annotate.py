import os
import subprocess
import argparse
import matplotlib.pyplot as plt
from shapely.wkt import loads
import tifffile as tiff
from config.config import Config
from utils.postgres_utils import PGConn
from utils.raster_utils import super_clip_interval

# FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "sentinel_annotation")

# Create directory if it doesn't exist
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
    pgconn = pgconn_obj.connection()
    table = config.setup_details["tables"]["villages"][0]

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_key", type=int, default=1298)
    parser.add_argument("-e", "--end_key", type=int, default=1900)
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
    
    QUADS_DIR = os.path.join(ROOT_DIR, args.interval, table['table'])
    print("Quads dir:", QUADS_DIR)

    for farm in poly_fetch_all:
        key = farm[0]
        images = []
        for i in range(interval_length):
            output_path = f'{os.path.dirname(os.path.realpath(__file__))}/temp/{i+1}_clipped.tif'
            multipolygon = loads(farm[1])
            super_clip_interval(QUADS_DIR, year, i+1, multipolygon, output_path)
            
            # Debug output
            print(f"Checking image at path: {output_path}")
            
            try:
                image = tiff.imread(output_path)
                images.append(image)
                print(f"Successfully loaded image {i+1} with shape: {image.shape}")
            except Exception as e:
                print(f"Error loading image at path {output_path}: {e}")

        # Display images and process user input
        i = 0
        while i < interval_length:
            print(f"{args.interval[:-2]}:", i+1)
            fig, axes = plt.subplots(nrows=interval_length//4, ncols=4, figsize=(12,8))
            axes = axes.flatten()
            
            for j, ax in enumerate(axes):
                if j < len(images):
                    rgb_image = images[j][:, :, :3]  # Assuming first three channels are RGB
                    ax.imshow(rgb_image)
                    ax.set_title(f'{args.interval[:-2]}: {j+1}')
                    if i == j:
                        ax.set_title(f'{args.interval[:-2]}: {j+1}', color='r')
                ax.axis('off')

            plt.suptitle(f"{table['key']}: {key}", fontsize=20)
            plt.tight_layout()
            plt.show()

            answer = input("Enter 'y' to save, 'n' to skip, 'd' to drop, 'f' to skip farm plot: ")
            if answer == "y" and i < len(images):
                file_name = f"{key}_{i+1}.tif"
                output_path = os.path.join(DATA_DIR, answer, file_name)
                try:
                    tiff.imwrite(output_path, images[i])
                    print(f"Saving to {output_path}")
                    i += 1
                except Exception as e:
                    print(f"Error saving image {i+1} to {output_path}: {e}")
            elif answer == "n" or answer == "d":
                i += 1
                if answer == "n":
                    output_path = os.path.join(DATA_DIR, "n", f"{key}_{i+1}.tif")
                else:
                    print("Skipping image due to clouds...")
            elif answer == "f":
                print("Skipping current farm plot...")
                break
            else:
                continue

            pgconn.commit()  # Commit after processing each image

    pgconn.commit()  # Final commit after processing all farm plots
