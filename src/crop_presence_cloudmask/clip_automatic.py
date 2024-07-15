import os
import argparse
import subprocess
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry
import numpy as np
from shapely.wkt import loads
from rasterio.mask import mask
from tqdm import tqdm
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *

## FILE PATHS
ROOT_DIR = os.path.abspath(subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip())
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Clip and compute hue features for villages and add to database') 
    parser.add_argument("-i", "--interval", type=str, help="Interval type: fortnightly, monthly, weekly", default="monthly")  


    args = parser.parse_args()
    args.interval = args.interval.lower()

    if args.interval == "fortnightly":
        interval_length = 24
    elif args.interval == "monthly":
        interval_length = 12
    elif args.interval == "weekly":
        interval_length = 48

    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()
    table = config.setup_details["tables"]["villages"][0]
    pgconn.commit()

    
    years = config.setup_details['months']['agricultural_years']
    year = years[0]
    print("Processing table", f'{table["schema"]}.{table["table"]}')
    
    for year in years:
        print(f"Processing Farmplots for agricultural year: {year}")
        for i in range(interval_length):
            sql_query = f"""
            alter table {table["schema"]}.{table["table"]}
            drop column if exists hue_mean_{year}_{args.interval}_{i+1};
            alter table {table["schema"]}.{table["table"]}
            drop column if exists hue_stddev_{year}_{args.interval}_{i+1};
            alter table {table["schema"]}.{table["table"]}
            drop column if exists ir_mean_{year}_{args.interval}_{i+1};
            alter table {table["schema"]}.{table["table"]}
            drop column if exists ir_stddev_{year}_{args.interval}_{i+1};
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
        
        for i in range(interval_length):
            sql_query = f"""
            alter table {table["schema"]}.{table["table"]}
            add column hue_mean_{year}_{args.interval}_{i+1} numeric;
            alter table {table["schema"]}.{table["table"]}
            add column hue_stddev_{year}_{args.interval}_{i+1} numeric;
            alter table {table["schema"]}.{table["table"]}
            add column ir_mean_{year}_{args.interval}_{i+1} numeric;
            alter table {table["schema"]}.{table["table"]}
            add column ir_stddev_{year}_{args.interval}_{i+1} numeric;
            """
            
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
        
        sql_query = f"""
        select
            {table["key"]},
            st_astext(st_transform({table["geom_col"]}, 4674))
        from
            {table["schema"]}.{table["table"]}
        where    
            {table["filter"]}
        """
        with pgconn.cursor() as curs:
            curs.execute(sql_query)
            poly_fetch_all = curs.fetchall()
        pgconn.commit()

        print(f"Total # Farmplots : {len(poly_fetch_all)}")

        QUADS_DIR = os.path.join(ROOT_DIR, f"{args.interval}", table['table'])
        print("Quads dir:", QUADS_DIR)

        
        for i in range(interval_length):

            print(f"# {args.interval[:-2].upper()}", i+1)
            cnt=0
            for poly in tqdm(poly_fetch_all, desc="Processing Farmplots"):
                output_path = os.path.join(CURRENT_DIR, "temp_clipped.tif")
                multipolygon = loads(poly[1])
                super_clip_interval(QUADS_DIR, year, i+1, multipolygon, output_path)
                tif_path = output_path
                hue_mean, hue_stddev, ir_mean, ir_stddev = compute_hue_ir_features(tif_path)
                if hue_mean==hue_stddev==ir_mean==ir_stddev==-1:
                    cnt+=1
                    
                sql_query = f"""
                update
                    {table["schema"]}.{table["table"]}
                set
                    hue_mean_{year}_{args.interval}_{i+1} = {hue_mean},
                    hue_stddev_{year}_{args.interval}_{i+1} = {hue_stddev},
                    ir_mean_{year}_{args.interval}_{i+1} = {ir_mean},
                    ir_stddev_{year}_{args.interval}_{i+1} = {ir_stddev}
                where
                    {table["key"]} = {poly[0]}
                """
                with pgconn.cursor() as curs:
                    curs.execute(sql_query)
            print(f"# Clouded Farmplots = {cnt}")
    
    pgconn.commit()