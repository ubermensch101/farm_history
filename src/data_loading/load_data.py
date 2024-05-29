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
    print("hi")
    
    return FarmplotLoading(config,pgconn, schema)

class FarmplotLoading:
    def __init__(self, config, psql_conn, schema):
        self.config = config
        self.psql_conn = psql_conn
        self.path = "/home/sameer/Downloads/Villages"
        # self.toggle = self.config.setup_details["data"]["toggle"]
        self.toggle = 0
        self.schema = schema
        self.temp = "temp_farmplots_tile_for_all" if schema != "" else "temp_farmplots_tile"
           
    def run(self):
        if self.path == "":
            print("Data path not set for farmplots")
            return
        srid = 32643
        table_name = "farmplots_amravati"
        for (root,dirs,files) in os.walk(self.path, topdown=True):
            
            try:
                village = os.path.basename(root).split('_')[0].lower()
                village = village.replace(' ','')
                if self.schema != "":
                    village = self.schema
                
                geom_files = []
                
                for file in files:
                    if file.endswith(".shp") or file.endswith(".kml"):
                        # village = file.split('_')[0].lower()
                        # village = village.replace(' ','')
                        print(file,village)
                        file_location = os.path.join(root,file)
                        geom_files.append(file_location)
                        
                if len(geom_files) == 0:
                    continue
                
                create_schema(self.psql_conn, village, delete_original= False)
                query = f"drop table if exists {village}.{table_name};"
                print("hi")
                with self.psql_conn.connection().cursor() as curs:
                    curs.execute(query)
                self.psql_conn.connection().commit()
                print("bye")
                
                for location in geom_files:
                    ogr2ogr_cmd = [
                        'ogr2ogr','-f','PostgreSQL','-t_srs',f'EPSG:{srid}',
                        'PG:dbname=' + self.psql_conn.details["database"] + ' host=' +
                            self.psql_conn.details["host"] + ' user=' + self.psql_conn.details["user"] +
                            ' password=' + self.psql_conn.details["password"],
                        location,
                        '-lco', 'OVERWRITE=YES',
                        '-lco', 'GEOMETRY_NAME=geom',
                        '-lco', 'schema=' + village, 
                        '-lco', 'SPATIAL_INDEX=GIST',
                        '-nlt', 'PROMOTE_TO_MULTI',
                        '-nln', self.temp
                    ]
                    subprocess.run(ogr2ogr_cmd) 
                    
                    sql = f"""
                        create table if not exists {village}.{table_name} 
                        as table {village}.{self.temp};
                        
                        insert into {village}.{table_name} 
                        select * from {village}.{self.temp};
                    """
                    with self.psql_conn.connection().cursor() as curr:
                        curr.execute(sql)
                        
                
                add_column(self.psql_conn, village+'.'+table_name, "gid", "serial")
                if self.schema != "":
                    with self.psql_conn.connection().cursor() as curr:
                        curr.execute(f"CREATE INDEX gist_index ON {village}.{table_name} USING GIST(geom);")
                
            except Exception as e:
                print("Error in loading farmplots for village",village)
                print(e)
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Description for parser")

    parser.add_argument("-p", "--path", help="Path to data",
                        required=False, default="")
    parser.add_argument("-t", "--toggle", help="0 for village path, 1 for taluka path, 2 for district path and 3 for state path",
                        required=False, default="")
    parser.add_argument("-f", "--farmpath", help="Path to farmplots",
                        required=False, default="")
    parser.add_argument("-s", "--schema", help="schema name where farmplot table will be added (used when combined farmplots is provided)",
                        required=False, default="")
    
    argument = parser.parse_args()
    path_to_data = argument.path
    toggle = argument.toggle
    path_to_farmplots = argument.farmpath
    schema = argument.schema
    
    fl = farmplotloading(path_to_farmplots, schema)
    fl.run()