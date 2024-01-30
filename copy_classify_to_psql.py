import argparse
import csv
import psycopg2
from config.config import Config
from utils.postgres_utils import PGConn

config = Config()
pgconn_obj = PGConn(config)
pgconn=pgconn_obj.connection()

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--append", type=int, required=True,
                    help="1 if you wish to append to the existing psql table \
                        and 0 if you wish to overwrite/create the psql table")
args = parser.parse_args()

if(args.append==0):
    sql_query = f"""
    drop table if exists pilot.cropping_presence_dagdagad;
    create table pilot.cropping_presence_dagdagad (
        gid integer,
        month text,
        year integer,
        crop_presence integer
    )
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)

pgconn.commit()

csv_file_path = "classify_output.csv"
with open(csv_file_path, 'r') as file, pgconn.cursor() as curs:
    reader = csv.reader(file)
    for row in reader:
        query = f"""
        insert into
            pilot.cropping_presence_dagdagad
        values
            ({row[0]}, '{row[1]}', {row[2]}, {row[3]})
        """
        curs.execute(query)

pgconn.commit()
        