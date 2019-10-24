# K Modes Clustering
# In this notebook we'll work on K Modes Clustering algorithm, to cluster a single categorical variable clustering task.  
# As usual, let's import the libraries first:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now, let's use a dummy dataset and see how it is:

data = pd.read_csv('data1.csv')

one_hot_df = data.copy()
for i,name in enumerate(data['Skill'].unique()):
    one_hot_df[name] = 0

def set_product(x):
    x[str(x['Skill'])] = 1
    return x

one_hot_df = one_hot_df.apply(set_product, axis=1)
one_hot_df = one_hot_df.groupby(['Individual']).sum()

from kmodes.kmodes import KModes
# define the k-modes model
km = KModes(n_clusters=3, init='Huang', n_init=4, verbose=1)
# fit the clusters to the skills dataframe
clusters = km.fit_predict(one_hot_df)
# get an array of cluster modes
kmodes = km.cluster_centroids_
shape = kmodes.shape
# For each cluster mode (a vector of "1" and "0")
# find and print the column headings where "1" appears.
# If no "1" appears, assign to "no-skills" cluster.
for i in range(shape[0]):
    if sum(kmodes[i,:]) == 0:
        print("\ncluster " + str(i) + ": ")
        print("no-skills cluster")
    else:
        print("\ncluster " + str(i) + ": ")
        cent = kmodes[i,:]
        for j in one_hot_df.columns[np.nonzero(cent)]:
            print(j)