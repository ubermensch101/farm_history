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

def plot_centroid_timeseries(centroid, month_labels, cluster_length, cluster_name):
    # Specify the directory to save plots
    output_dir = "/home/rahul/farm_history/src/analysis/cluster_plots"
    
    # Ensure the directory exists or create it if it doesn't
    os.makedirs(output_dir, exist_ok=True)
    
    # Plotting the centroid time series
    plt.figure(figsize=(20, 12))
    plt.plot(centroid, label='Cluster Centroid', linewidth=3)
    plt.title(f'Centroid Time Series Plot (Cluster Length: {cluster_length})')
    plt.xlabel('Month')
    plt.ylabel('Crop Presence Probability')
    plt.xticks(np.arange(len(month_labels)), month_labels)
    plt.grid(True)
    plt.legend()
    
    # Save the plot as an image file in the specified directory
    plt.savefig(os.path.join(output_dir, f"{cluster_name}.png"))
    plt.show()

def annotate_clusters(clusters, ids, unwanted_clusters):
    annotated_labels = {
        "a": "kharif_rabi",
        "b": "short_kharif",
        "c": "long_kharif",
        "p": "perennial",
        "z": "zaid",
        "n": "no_crop",
        "w": "weed",
        "m": "mystery",
        "1": "mystery1",
        "2": "mystery2",
        "3": "mystery3",
        "4": "mystery4",
        "5": "mystery5",
        "6": "mystery6",
        "7": "mystery7",
        "8": "mystery8",
        "9": "mystery9",
        "10": "mystery10",
        "11": "mystery11",
        "12": "mystery12",
        "13": "mystery13",
    }
    labeled_clusters = {}

    for cluster in np.unique(clusters):
        if cluster not in unwanted_clusters:
            # Calculate the length of the current cluster
            cluster_length = np.sum(clusters == cluster)
            
            # Name the cluster according to its label
            cluster_name = f"mystery{cluster}"

            # Save and display the centroid time series plot for the current cluster
            plot_centroid_timeseries(centroids[cluster], to_display, cluster_length, cluster_name)
            
            # Prompt the user to annotate the cluster
            print(f"Annotate Cluster {cluster} with one of the following labels:")
            print("a : kharif & rabi")
            print("b : short kharif")
            print("c : long kharif")
            print("p : perennial")
            print("z : zaid")
            print("n : no_crop")
            print("w : weed")
            print("m : mystery")
            print("1 : mystery1")
            print("2 : mystery2")
            print("3 : mystery3")
            print("4 : mystery4")
            print("5 : mystery5")
            print("6 : mystery6")
            print("7 : mystery7")
            print("8 : mystery8")
            print("9 : mystery9")

            label = input("Enter the label (press Enter to skip): ").lower()

            # Skip the cluster if no label is entered
            if label == "":
                print(f"Skipping Cluster {cluster}")
                continue

            # Store the label for the current cluster
            labeled_clusters[cluster] = annotated_labels[label]
            
            # Update the database with the predicted labels for the corresponding IDs in the cluster
            ids_in_cluster = ids[clusters == cluster]
            update_query = f"""
            UPDATE {table["schema"]}.{table["table"]}
            SET kcluster = '{annotated_labels[label]}'
            WHERE id IN ({', '.join(map(str, ids_in_cluster))})
            """
            with pgconn.cursor() as curs:
                curs.execute(update_query)
                
    return labeled_clusters

if __name__ == "__main__":
    config = Config()
    pgconn_obj = PGConn(config)
    pgconn = pgconn_obj.connection()

    table = config.setup_details["tables"]["villages"][0]
    
    # Step 1: Data Retrieval
    crop_presence_data = []
    ids = []  # List to store id
    
    # Fetch crop presence probabilities and id for all locations
    columns = ""
    months_data = ['01', '02', '03', '04', '05', '06',
                   '07', '08', '09', '10', '11', '12']
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
    SELECT {columns}, id
    FROM {table["schema"]}.{table["table"]}
    where kcluster='mystery9'
    """

    with pgconn.cursor() as curs:
        print(sql_query)
        curs.execute(sql_query)
        records = curs.fetchall()
        for record in records:
            crop_presence_data.append(record[:-1])  # Exclude the last column (id)
            ids.append(record[-1])  # Extract id

    crop_presence_data = np.array(crop_presence_data)
    ids = np.array(ids)

    # Step 2: Preprocessing (if needed)

    # Step 3: Time Series K-Means Clustering
    k = 1  # Number of clusters
    ts_kmeans = TimeSeriesKMeans(n_clusters=k, metric="euclidean", verbose=1, random_state=0)
    clusters = ts_kmeans.fit_predict(crop_presence_data)

    # Get centroids
    centroids = ts_kmeans.cluster_centers_
    cluster_counts = np.bincount(clusters)
    
    # Specify the cluster(s) you don't want to keep
    unwanted_clusters = []  # Example: clusters 1, 3, and 5
    
    # List to store id of elements in deleted clusters
    del_ids = []
    cluster_counts = np.bincount(clusters)
    print(cluster_counts)
    # Remove unwanted clusters
    for cluster in unwanted_clusters:
        indices = np.where(clusters == cluster)[0]  # Find indices of elements in unwanted clusters
        del_ids.extend(ids[indices])  # Store id of elements in deleted clusters
        clusters = np.delete(clusters, indices)  # Remove elements in unwanted clusters
        crop_presence_data = np.delete(crop_presence_data, indices, axis=0)  # Remove corresponding data
    
    # Step 4: Annotation and Database Update
    labeled_clusters = annotate_clusters(clusters, ids, unwanted_clusters)
