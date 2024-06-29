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

def farmplotloading(path_to_farmplots = "", schema = ""):
    config = Config()
    if path_to_farmplots != "":
        if "data" not in config.setup_details:
            config.setup_details["data"] = {}
        config.setup_details["data"]["farmplots_path"] = path_to_farmplots
    pgconn = PGConn(config)
    return FarmplotLoading(config,pgconn, schema)

class FarmplotLoading:
    def __init__(self, config, psql_conn, schema):
        self.config = config
        self.psql_conn = psql_conn
        self.path = config.setup_details["data"]["farmplots_path"]
        self.schema = schema
        self.temp = "temp_farmplots_tile_for_all" if schema != "" else "temp_farmplots_tile"
           
    def run(self):
        if self.path == "":
            print("Data path not set for farmplots")
            return
        srid = 4326
        for (root,dirs,files) in os.walk(self.path, topdown=True):
            try:
                for file in files:
                    if file.endswith(".shp") or file.endswith(".kml"):
                        village = file.split('.')[0].lower()
                        village = village.replace(' ','')
                        print(f"Adding village  {village}")
                        file_location = os.path.join(root,file)
                        # geom_files.append(file_location)
                        create_schema(self.psql_conn, self.schema, delete_original= False)
                        query = f"drop table if exists {self.schema}.{village};"
                        with self.psql_conn.connection().cursor() as curs:
                            curs.execute(query)
                            print(f"Dropping table {self.schema}.{village} if exist")
                        self.psql_conn.connection().commit()

                        ogr2ogr_cmd = [
                        'ogr2ogr','-f','PostgreSQL','-t_srs',f'EPSG:{srid}',
                        'PG:dbname=' + self.psql_conn.details["database"] + ' host=' +
                            self.psql_conn.details["host"] + ' user=' + self.psql_conn.details["user"] +
                            ' password=' + self.psql_conn.details["password"],
                        file_location,
                        '-lco', 'OVERWRITE=YES',
                        '-lco', 'schema=' + self.schema, 
                        '-lco', 'SPATIAL_INDEX=GIST',
                        '-nlt', 'PROMOTE_TO_MULTI',
                        '-nln', village
                    ]
                    subprocess.run(ogr2ogr_cmd) 

 
                    
                        
                # add_column(self.psql_conn, village+'.'+table_name, "gid", "serial")
                # if self.schema != "":
                #     with self.psql_conn.connection().cursor() as curr:
                #         curr.execute(f"CREATE INDEX gist_index ON {village}.{table_name} USING GIST(geom);")
                
            except Exception as e:
                print("Error in loading farmplots for village",village)
                print(e)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Description for parser")

    parser.add_argument("-p", "--path", help="Path to combined farmplots for villages", required=True)
    parser.add_argument("-s", "--schema", help="schema name where farmplot table will be added (used when combined farmplots is provided)",
                        required=False, default="public")
    
    argument = parser.parse_args()
    path_to_farmplots = argument.path
    schema = argument.schema
    fl = farmplotloading(path_to_farmplots, schema)
    fl.run()