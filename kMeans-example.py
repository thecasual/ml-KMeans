import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd


# Given np array return X clusters of kmeans
# No labels required - unsupervised learning
# Credit - https://pythonprogramming.net/flat-clustering-machine-learning-python-scikit-learn/
# Maybe anomaly detection??






# Reading in test data
file = "data.csv"
feature_cols = ['Pclass', 'Fare']

# Uncomment here to see array from link above
#X = np.array([[1, 2],[5, 8],[1.5, 1.8],[8, 8],[1, 0.6],[9, 11]])
X = pd.read_csv(file, skipinitialspace=True, usecols=feature_cols).to_numpy()

# Put into 5 buckets
kmeans = KMeans(n_clusters=5)

# Magic
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)
colors = ["g.","r.","b.","c.","y."]


# Labels is the unsupervised label that is being assigned..
# Label 0 = Green
# Label 1 = Red...etc

# Plot the data provided - no magic here
for i in range(len(X)):
    print("coordinate : {}\nlabel: {}".format(X[i], labels[i]))
    
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)


# Plot the "centroids" which is essentially the averaged value
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "x", s=150, linewidths= 5, zorder = 10)
plt.show()

# Found anomaly where someone paid $512 for a ticket!!