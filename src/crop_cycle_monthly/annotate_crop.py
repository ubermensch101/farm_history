import os
import shutil
import subprocess
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *


## Utils
def copy_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    files = os.listdir(input_directory)

    for file in files:
        if file.endswith(".tif"):
            input_file_path = os.path.join(input_directory, file)
            if os.path.isfile(input_file_path):
                output_file_path = os.path.join(output_directory, file)
                shutil.copy2(input_file_path, output_file_path)


## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
DATA_DIR = os.path.join(ROOT_DIR, "data", "crop_cycle")
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TEMP_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
ANNOTATION_FILE = os.path.join(CURRENT_DIR, "annotations", "crop_data_after_final.csv")
crop_classes = ["kharif_rabi", "short_kharif", "long_kharif","perennial","zaid","no_crop","weed"]
label2dir = {
    "a": "kharif_rabi",
    "b": "short_kharif",
    "c": "long_kharif",
    "p": "perennial",
    "z": "zaid",
    "n": "no_crop",
    "w": "weed"
}

if __name__ == "__main__":
    # Create directory if doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            for crop_class in crop_classes:
                os.makedirs(os.path.join(DATA_DIR, crop_class))

        except OSError as e:
            print(f"Error creating directory '{DATA_DIR}': \n {e}")

    text = """
        Annotation Instructions: \n\n
        Input format: \n\n
        a : kharif & rabi \n
        b : short kharif\n
        c : long kharif\n
        p : perennial\n
        z : zaid\n
        n : no crop\n
        w : weed\n
    """
    # Create directory if doesn't exist
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            for crop_class in crop_classes:
                os.makedirs(os.path.join(DATA_DIR, crop_class))

        except OSError as e:
            print(f"Error creating directory '{DATA_DIR}': \n {e}")
    print(text)

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    months_data = ["01","02","03","04","05","06",
                   "07","08","09","10","11","12",]
    months_names = ["jan","feb","mar","apr","may","jun",
                    "jul", "aug", "sep","oct","nov","dec",]
    study_months = config.setup_details["months"]["agricultural_months"]
    sql_query = f"""
    select
        {table["key"]},
        st_astext(st_transform({table["geom_col"]}, 3857))
    from
        {table["schema"]}.{table["table"]}
    where
        {table['filter']}
    order by
        random()
    limit
        10
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        poly_fetch_all = curs.fetchall()

    for poly in poly_fetch_all:
        key = poly[0]
        print(f"Processing on farm_id: {key}")
        images = []
        for month in study_months:
            output_path = f"{os.path.dirname(os.path.realpath(__file__))}/temp/{months_names[month[1]]}_{month[0]}_clipped.tif"
            multipolygon = loads(poly[1])
            super_clip(
                "quads", month[0], months_data[month[1]], multipolygon, output_path
            )
            images.append(np.array(Image.open(output_path)))
            bands = calculate_average_color(output_path)

            ## Get prediction using bands & Add to csv

        columns = ""

        if not check_column_exists(pgconn_obj, table["schema"], table["table"], "crop_cycle_22_23"):
            add_column(pgconn_obj, f"{table['schema']}.{table['table']}", "crop_cycle_22_23", "text")
        
        for month in study_months:
            columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"
    
        sql_query = f"""
        select
            {columns}
            crop_cycle_22_23
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {key}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            record = curs.fetchall()[0]

        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 8))
        axes = axes.flatten()

        for i, (ax, image, month) in enumerate(zip(axes, images, study_months)):
            ax.imshow(image)
            ax.set_title(
                f"{months_names[month[1]]}-{month[0]} prob: {round(record[i], 3)}"
            )

        plt.suptitle(f"{table['key']}: {key}; cropping pattern: {record[-1]}", fontsize=20)
        plt.tight_layout()
        plt.show()

        plt.pause(1)
        plt.close()

        label = input()

        if label.lower() in ["a", "b", "c", "p","z","n","w"]:
            output_path = os.path.join(
                DATA_DIR, label2dir[label.lower()], f"{table['table']}_{key}"
            )
            print(output_path)
            copy_files(TEMP_DIR, output_path)
