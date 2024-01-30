from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import Config
from utils.postgres_utils import PGConn

config = Config()
pgconn_obj = PGConn(config)
pgconn=pgconn_obj.connection()

query = """
select
    crop_presence,
    red,
    green,
    blue,
    nir
from
    pilot.cropping_presence_dagdagad as c
join
    pilot.average_bands_dagdagad as a
on
    c.gid = a.gid
and
    c.month = a.month
and
    c.year = a.year
"""
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

print("dataset size:", len(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

predictions = [1 if x else 0 for x in [item >= 0.025 for item in GRVI]]
predictions = model.predict(X_test)
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))

pgconn.commit()