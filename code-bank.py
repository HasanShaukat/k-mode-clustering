import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Now, let's load the bank dataset:

bank = pd.read_csv('bank/bank-full.csv', delimiter=';')

from kmodes.kmodes import KModes
# define the k-modes model
km = KModes(n_clusters=5, init='Huang', n_init=1, verbose=1)
# fit the clusters to the bank dataset
clusters = km.fit_predict(bank)
# get an array of cluster modes
kmodes = km.cluster_centroids_
shape = kmodes.shape
for i in range(shape[0]):
    print(kmodes[i,:])
