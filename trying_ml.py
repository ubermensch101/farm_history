import copy
import json
import math
import os
import pathlib
import time
import psycopg2
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rasterio
from shapely.wkt import loads
from rasterio.mask import mask
import cv2

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

class PGConn:
    def __init__(self, host, port, dbname, user=None, passwd=None):
        self.host = host
        self.port = port
        self.dbname = dbname
        if user is not None:
            self.user = user
        else:
            self.user = ""
        if passwd is not None:
            self.passwd = passwd
        else:
            self.passwd = ""
        self.conn = None

    def connection(self):
        """Return connection to PostgreSQL.  It does not need to be closed
        explicitly.  See the destructor definition below.

        """
        if self.conn is None:
            conn = psycopg2.connect(dbname=self.dbname,
                                    host=self.host,
                                    port=str(self.port),
                                    user=self.user,
                                    password=self.passwd)
            self.conn = conn
            
        return self.conn

pgconn_obj = PGConn(
    "localhost",
    5432,
    "dolr",
    "sameer",
    "swimgood"
)
    
pgconn=pgconn_obj.connection()

query = "select crop_presence, red, green, blue, nir from pilot.cropping_presence_dagdagad as c join pilot.average_bands_dagdagad as a on c.gid = a.gid and c.month = a.month"
with pgconn.cursor() as curs:
    curs.execute(query)
    rows = curs.fetchall()

NDVI = []
GRVI = []
X = [[float(band) for band in item[1:5]] for item in rows]
for i in range(len(X)):
    rgbn = X[i]
    NDVI.append((rgbn[3] - rgbn[0])/(rgbn[3] + rgbn[0]))
    GRVI.append((rgbn[1] - rgbn[0])/(rgbn[1] + rgbn[0]))
Y = [item[0] for item in rows]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = [1 if x else 0 for x in [item >= 0.025 for item in GRVI]]
predictions = model.predict(X_test)
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

pgconn.commit()