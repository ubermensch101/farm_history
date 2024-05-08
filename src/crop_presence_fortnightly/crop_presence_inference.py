import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import *
from utils.postgres_utils import *

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    for i in range(24):
        if not check_column_exists(pgconn_obj, table["schema"], table["table"],
                               f'crop_presence_{i+1}'):
            sql_query = f"""
            alter table
                {table["schema"]}.{table["table"]}
            add column
                crop_presence_{i+1} numeric
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)

    with open(f'{os.path.dirname(os.path.realpath(__file__))}/crop_presence_detector.pkl', 'rb') as file:
        model = pickle.load(file)

    sql_query = f"""
    select
        {table["key"]}
    from
        {table["schema"]}.{table["table"]}
    where
        {table["filter"]}
    order by
        {table["key"]}
    """
    with pgconn.cursor() as curs:
        curs.execute(sql_query)
        rows = curs.fetchall()
    keys = [item[0] for item in rows]

    for i in range(24):
        print("fortnight:", i+1)
        for key in keys:
            sql_query = f"""
            select
                hue_mean_{i+1},
                hue_stddev_{i+1}
            from
                {table["schema"]}.{table["table"]}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
                row = curs.fetchall()[0]
            bands = [[float(item) for item in row]]
            crop_presence = model.predict_proba(bands)[0][1]
            
            sql_query = f"""
            update
                {table["schema"]}.{table["table"]}
            set
                crop_presence_{i+1} = {crop_presence}
            where
                {table["key"]} = {key}
            """
            with pgconn.cursor() as curs:
                curs.execute(sql_query)
    pgconn.commit()