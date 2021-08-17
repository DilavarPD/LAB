import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.csv")

x = iris[["sepal_length","sepal_width"]]
ML = KMeans(n_clusters=3,max_iter=2)
ML = ML.fit(x)

centers = ML.cluster_centers_
labels = ML.labels_

xaxis = iris["sepal_length"]
yaxis = iris["sepal_width"]
plt.scatter(x=xaxis,y=yaxis,c=labels,cmap="rainbow")
plt.show()