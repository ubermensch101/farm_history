import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psycopg2

# Function to fetch count from the database
def fetch_count_from_database(crop_label, kcluster_label):
    try:
        conn = psycopg2.connect(
            dbname="telangana_villages",
            user="postgres",
            password="postgres",
            host="localhost",
            port="5432"
        )
    except psycopg2.Error as e:
        print("Unable to connect to the database.")
        print(e)
        return 0

    try:
        cursor = conn.cursor()
        query = f"""
            SELECT COUNT(*)
            FROM hunsa_cleaned
            WHERE crop_cycle_22_23 = '{crop_label}' AND kcluster = '{kcluster_label}'
        """
        cursor.execute(query)
        count = cursor.fetchone()[0]
        return count
    except psycopg2.Error as e:
        print("Error fetching data from the database.")
        print(e)
        return 0
    finally:
        if conn is not None:
            conn.close()

# Define the labels
labels = ["kharif_rabi", "short_kharif", "long_kharif", "perennial", "zaid", "weed", "no_crop", "mystery"]

# Initialize an empty confusion matrix with zeros
conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

# Fetch count for each combination and fill the matrix
for i, crop_label in enumerate(labels):
    for j, kcluster_label in enumerate(labels):
        conf_matrix[i, j] = fetch_count_from_database(crop_label, kcluster_label)

# Normalize the confusion matrix (divide each column by its sum)
column_sums = conf_matrix.sum(axis=0, keepdims=True)
conf_matrix_normalized = np.where(column_sums > 0, conf_matrix / column_sums, 0)

# Create DataFrame with all labels
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
conf_matrix_normalized_df = pd.DataFrame(conf_matrix_normalized, index=labels, columns=labels)

# Plotting both confusion matrices side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot the confusion matrix with counts
im1 = ax1.imshow(conf_matrix_df, cmap='Blues', interpolation='nearest')

# Add color bar to the first plot
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label('Count')

# Add annotations to the first plot
for i in range(conf_matrix_df.shape[0]):
    for j in range(conf_matrix_df.shape[1]):
        ax1.text(j, i, f"{conf_matrix_df.iloc[i, j]}",
                ha="center", va="center",
                color="white" if conf_matrix_df.iloc[i, j] > conf_matrix_df.max().max() / 2 else "black")

# Set labels and ticks for the first plot
ax1.set_title('Confusion Matrix (Counts)')
ax1.set_xlabel('kcluster')
ax1.set_ylabel('crop_cycle_22_23')
ax1.set_xticks(np.arange(len(labels)))
ax1.set_yticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, rotation=45)
ax1.set_yticklabels(labels)

# Plot the normalized confusion matrix
im2 = ax2.imshow(conf_matrix_normalized_df, cmap='Blues', interpolation='nearest')

# Add color bar to the second plot
cbar2 = fig.colorbar(im2, ax=ax2)
cbar2.set_label('Percentage')

# Add annotations to the second plot
for i in range(conf_matrix_normalized_df.shape[0]):
    for j in range(conf_matrix_normalized_df.shape[1]):
        ax2.text(j, i, f"{conf_matrix_normalized_df.iloc[i, j]*100:.2f}%",
                ha="center", va="center",
                color="white" if conf_matrix_normalized_df.iloc[i, j] > 0.5 else "black")

# Set labels and ticks for the second plot
ax2.set_title('Confusion Matrix (Normalized)')
ax2.set_xlabel('kcluster')
ax2.set_ylabel('crop_cycle_22_23')
ax2.set_xticks(np.arange(len(labels)))
ax2.set_yticks(np.arange(len(labels)))
ax2.set_xticklabels(labels, rotation=45)
ax2.set_yticklabels(labels)

plt.tight_layout()
plt.show()
