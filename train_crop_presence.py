import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import Config
from utils.postgres_utils import PGConn

if __name__=='__main__':
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn=pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]

    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    ml_months = config.setup_details["months"]["agricultural_months"]
    columns = ""
    for month in ml_months:
        columns += f"""
            {months_names[month[1]]}_{month[0]}_red,
            {months_names[month[1]]}_{month[0]}_green,
            {months_names[month[1]]}_{month[0]}_blue,
            {months_names[month[1]]}_{month[0]}_nir,
            """

    # Getting annotations
    annotations = []
    with open('annotations.csv') as file:
        for line in file.readlines():
            annotations.append([int(item) for item in line.strip().split(',')])

    # Gathering data for training the model
    X = []
    Y = []
    for row in annotations:
        query = f"""
        select
            {columns}
            {table["key"]}
        from
            {table["schema"]}.{table["table"]}
        where
            {table["key"]} = {row[0]}
        """
        with pgconn.cursor() as curs:
            curs.execute(query)
            bands = curs.fetchall()[0]
        for i in np.linspace(0,11,num=12):
            X.append((
                float(bands[int(4*i)]),
                float(bands[int(4*i+1)]),
                float(bands[int(4*i+2)]),
                float(bands[int(4*i)+3])
            ))
            Y.append(row[int(i+1)])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    print(classification_report(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))

    with open('crop_presence_detector.pkl', 'wb') as file:
        pickle.dump(model, file)