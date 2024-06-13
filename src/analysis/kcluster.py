import os
import shutil
import subprocess
import argparse
import cv2
import matplotlib.pyplot as plt
from shapely.wkt import loads
import pandas as pd
import numpy as np
from config.config import Config
from utils.postgres_utils import *
from utils.raster_utils import *
from tslearn.clustering import TimeSeriesKMeans

def plot_centroid_timeseries(centroids, labels, month_labels):
    plt.figure(figsize=(10, 6))
    
    # Calculate the number of elements in each cluster
    cluster_counts = np.bincount(labels)
    
    # Calculate the maximum number of elements in any cluster
    max_num_elements = max(cluster_counts)
    
    for i, centroid in enumerate(centroids):
        # Normalize the count to range [1, 3] for line width
        line_width = (cluster_counts[i] / max_num_elements)*3
        plt.plot(centroid, label=f'Cluster {i + 1}', linewidth=line_width)

    plt.title('Centroid Time Series Plot')
    plt.xlabel('Month')
    plt.ylabel('Crop Presence Probability')
    plt.xticks(np.arange(len(month_labels)), month_labels)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    
    # Step 1: Data Retrieval
    crop_presence_data = []
    ogc_fids = []  # List to store ogc_fid
    
    # Fetch crop presence probabilities and ogc_fid for all locations
    columns = ""
    months_data = ['01','02','03','04','05','06',
                   '07','08','09','10','11','12']
    months_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    cycle_months = config.setup_details["months"]["agricultural_months"]
    to_display = ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan', 'feb', 'mar', 'apr']
    columns = ""
    for month in cycle_months:
        columns += f"{months_names[month[1]]}_{month[0]}_crop_presence,"
    columns = columns[:-1]
    print(columns)
    sql_query = f"""
    SELECT {columns}, ogc_fid
    FROM {table["schema"]}.{table["table"]}
    WHERE (crop_cycle_22_23 = 'kharif_rabi')
    """

    with pgconn.cursor() as curs:
        print(sql_query)
        curs.execute(sql_query)
        records = curs.fetchall()
        for record in records:
            crop_presence_data.append(record[:-1])  # Exclude the last column (crop cycle)
            ogc_fids.append(record[-1])  # Extract ogc_fid

    crop_presence_data = np.array(crop_presence_data)
    ogc_fids = np.array(ogc_fids)

    # Step 2: Preprocessing (if needed)

    # Step 3: Time Series K-Means Clustering
    k = 4  # Number of clusters
    ts_kmeans = TimeSeriesKMeans(n_clusters=k, metric="euclidean", verbose=1, random_state=0)
    clusters = ts_kmeans.fit_predict(crop_presence_data)

    # Get centroids
    centroids = ts_kmeans.cluster_centers_
    cluster_counts = np.bincount(clusters)
    
    # Specify the cluster(s) you don't want to keep
    unwanted_clusters = []  # Example: clusters 1, 3, and 5
    
    # List to store ogc_fid of elements in deleted clusters
    del_ogc = []
    cluster_counts = np.bincount(clusters)
    print(cluster_counts)
    # Remove unwanted clusters
    for cluster in unwanted_clusters:
        indices = np.where(clusters == cluster)[0]  # Find indices of elements in unwanted clusters
        del_ogc.extend(ogc_fids[indices])  # Store ogc_fid of elements in deleted clusters
        clusters = np.delete(clusters, indices)  # Remove elements in unwanted clusters
        crop_presence_data = np.delete(crop_presence_data, indices, axis=0)  # Remove corresponding data
    
    # Step 4: Visualization - Centroid Time Series Plot
    plot_centroid_timeseries(centroids, clusters, to_display)
