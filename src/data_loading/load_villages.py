"""
Utility to add farmplots for villages to the database.
Inputs: 
    1. Path to a folder containing shapefiles or kml files for each village with name of the village as the file name.
    2. Schema name where the farmplot tables will be added. Default is public.
Output: Farmplots for each village are added to the database.

"""

from config import *
from utils import *
import argparse
import os
import subprocess



def load_village(pgconn, file_path, schema, village, srid=4326):
    ogr2ogr_cmd = [
        'ogr2ogr','-f','PostgreSQL','-t_srs',f'EPSG:{srid}',
        'PG:dbname=' + pgconn.details["database"] + ' host=' +
        pgconn.details["host"] + ' user=' + pgconn.details["user"] +
        ' password=' + pgconn.details["password"],
        file_path,
        '-lco', 'OVERWRITE=YES',
        '-lco', 'schema=' + schema, 
        '-lco', 'SPATIAL_INDEX=GIST',
        '-nlt', 'PROMOTE_TO_MULTI',
        '-nln', village.lower()
    ]
    subprocess.run(ogr2ogr_cmd) 

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Utility to add farmplots for villages to the database.")
    parser.add_argument("-p", "--path", help="Path to combined farmplots for villages", required=True)
    parser.add_argument("-s", "--schema", help="schema name where farmplot table will be added (used when combined farmplots is provided)", default="public")
    args = parser.parse_args()
    
    path_to_farmplots = args.path
    schema = args.schema

    config = Config()
    pgconn = PGConn(config)

    if path_to_farmplots == "":
        print("Data path not set for farmplots")
        exit()
    

    srid = 4326
    for (root,dirs,files) in os.walk(path_to_farmplots, topdown=True):
        try:
            for file in files:
                if file.endswith(".shp") or file.endswith(".kml"):
                    village = file.split('.')[0].lower()
                    village = village.replace(' ','')
                    print(f"Adding village  {village}")
                    file_path = os.path.join(root,file)
                    create_schema(pgconn, schema)
                    query = f"drop table if exists {schema}.{village};"

                    with pgconn.connection().cursor() as curs:
                        curs.execute(query)
                        print(f"Dropping table {schema}.{village} if exist")

                    load_village(pgconn, file_path, schema, village, srid)

        except Exception as e:
            print("Error in loading farmplots for village",village)
            print(e)
