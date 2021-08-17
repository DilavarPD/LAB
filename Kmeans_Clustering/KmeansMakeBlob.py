
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


dataset = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8, random_state=50)
points = dataset[0]
k_means = KMeans(n_clusters=4, n_init=10)

k_means.fit(points)
clusters = k_means.cluster_centers_
y = k_means.fit_predict(points)
plt.scatter(points[y == 0,0],points[y == 0,1],s =50, color = 'red')
plt.scatter(points[y == 1,0],points[y == 1,1],s =50, color = 'blue')
plt.scatter(points[y == 2,0],points[y == 2,1],s =50, color = 'yellow')
plt.scatter(points[y == 3,0],points[y == 3,1],s =50, color = 'green')
plt.show()