import csv
import numpy as np
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

def load_data(filepath):
    # takes in a string with a path to a CSV file and returns the data points as a list of dicts
    data = []  # list of dictionaries
    with open(filepath, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:  # each row in the dataset is a dictionary with the column headers as keys and the row elements as values
            data.append(dict(row))  # convert to dict

    return data

def calc_features(row):
    # takes in one-row dict from the data loaded from the previous function
    # then calculates the corresponding feature vector for that country as specified above,
    # and returns it as a NumPy array of shape (6,).
    # The dtype of this array should be float64.

    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])

    return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

def hac(features):
    #1. Number each of your starting data points from 0 to n − 1. These are their original cluster numbers.
    n = len(features)
    clusters = {i: set([i]) for i in range(n)}

    #2. Create an (n − 1) × 4 array or list. Iterate through this array/list row by row.
    Z = np.zeros((n - 1, 4))

    # when n is large you should create a distance matrix at the beginning of the function to avoid having to recalculate the distances between points
    distances = distance_matrix(features, features)
    np.fill_diagonal(distances, np.inf)
    
    new_cluster_index = n  # start new cluster indices from n

    # the original countries are considered clusters indexed by 0, . . . , n − 1,
    # and the cluster constructed in the ith iteration (i ≥ 1) of the algorithm has cluster index (n − 1) + i
    for i in range(n - 1):
        min_distance = np.inf
        x, y = None, None

        # iterate over each pair of clusters
        for j in clusters:
            for k in clusters:
                if j == k:
                    continue
                # compute the distance between clusters j and k
                current_distance = -np.inf
                # Tie Breaking
                for a in clusters[j]:
                    for b in clusters[k]:
                        if distances[a][b] > current_distance:
                            current_distance = distances[a][b]
                if current_distance < min_distance:
                    min_distance = current_distance
                    x, y = j, k
                elif current_distance == min_distance:
                    if j < x or (j == x and k < y):
                        x, y = j, k

        # merge clusters x and y
        merged_cluster = clusters[x].union(clusters[y])
        
        # Determine which clusters are closest and put their numbers into the first and second elements of the row, Z[i, 0] and Z[i, 1].
        # The first element listed, Z[i, 0] should be the smaller of the two cluster indexes.
        Z[i, 0], Z[i, 1] = x, y
        Z[i, 2] = min_distance # The complete-linkage distance between the two clusters goes into the third element of the row, Z[i, 2]
        Z[i, 3] = len(merged_cluster) # The total number of countries in the new cluster goes into the fourth element, Z[i, 3]

        # delete old clusters x and y, and assign the merged cluster a new index
        del clusters[x]
        del clusters[y]
        clusters[new_cluster_index] = merged_cluster
        new_cluster_index += 1  # increment for the next new cluster

    #3. Before returning the data structure, convert it into a NumPy array if it isn’t one already.
    return np.array(Z)

def fig_hac(Z, names):
    # visualizes the hierarchical agglomerative clustering on the country’s feature representation

    fig = plt.figure(figsize=(10, 7))  # initialize a figure
    dendrogram(Z, labels=names, leaf_rotation=90)  # plot is likely to cut off the x labels.
    plt.tight_layout()
    plt.show()

    return fig

def normalize_features(features):
    # takes a list of feature vectors and computes the normalized values.
    # The output should be a list of normalized feature vectors in the same format as the input.

    feature_matrix = np.array(features)  # original value
    # np.mean(arr, axis = None): compute the arithmetic mean (average) of the given data (array elements) along the specified axis.
    means = np.mean(feature_matrix, axis=0)  # column’s mean
    # np.std(arr, axis = None): compute the standard deviation of the given data (array elements) along the specified axis(if any)..
    standard_deviation = np.std(feature_matrix, axis=0)  # column’s standard deviation
    # Use the equation to calculate the normalized feature values for each data point: (𝑥−μ) / σ
    normalized_matrix = (feature_matrix - means) / standard_deviation

    # return value should be in a format of list of the feature vectors
    return [row for row in normalized_matrix]