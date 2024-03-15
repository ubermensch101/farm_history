import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from config.config import Config
from utils.postgres_utils import PGConn
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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
                for i in np.linspace(0, 11, num=12):
                    if bands[int(4 * i)] is not None:
                        X.append([
                            float(bands[int(4 * i)]),
                            float(bands[int(4 * i + 1)]),
                            float(bands[int(4 * i + 2)]),
                            float(bands[int(4 * i) + 3])
                        ])
                        Y.append(row[int(i + 1)])

    # Convert X and Y to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # Reshape X to match the input shape of the CNN
    X = X.reshape(-1, len(ml_months), 4, 1)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(len(ml_months), 4, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Save the trained model
    #model.save('crop_presence_detector.h5')
    with open('crop_presence_detector.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Gathering data for training the model
    # X = []
    # Y = []
    # for row in annotations:
    #     query = f"""
    #     select
    #         {columns}
    #         {table["key"]}
    #     from
    #         {table["schema"]}.{table["table"]}
    #     where
    #         {table["key"]} = {row[0]}
    #     """
    #     with pgconn.cursor() as curs:
    #         curs.execute(query)
    #         bands = curs.fetchall()[0]
    #     for i in np.linspace(0,11,num=12):
    #         X.append((
    #             float(bands[int(4*i)]),
    #             float(bands[int(4*i+1)]),
    #             float(bands[int(4*i+2)]),
    #             float(bands[int(4*i)+3])
    #         ))
    #         Y.append(row[int(i+1)])

    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # model = LogisticRegression()
    # model.fit(X_train, Y_train)

    # predictions = model.predict(X_test)
    # print(classification_report(Y_test, predictions))
    # print(confusion_matrix(Y_test, predictions))

    # with open('crop_presence_detector.pkl', 'wb') as file:
    #     pickle.dump(model, file)

        #Another task for you: shifting the crop presence detector from the average colour based one to a CNN based one. You'll have to deal with the variable farmplot size to make this work.
