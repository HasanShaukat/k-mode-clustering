# K Modes Clustering
# In this notebook we'll work on K Modes Clustering algorithm, to cluster a single categorical variable clustering task.  
# As usual, let's import the libraries first:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now, let's load the bank dataset:

bank = pd.read_csv('bank/bank-full.csv', delimiter=';')

from kmodes.kmodes import KModes
# define the k-modes model
km = KModes(n_clusters=5, init='Huang', n_init=1, verbose=1)
# fit the clusters to the skills dataframe
clusters = km.fit_predict(bank)
# get an array of cluster modes
kmodes = km.cluster_centroids_
shape = kmodes.shape
for i in range(shape[0]):
    print(kmodes[i,:])
