import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
# data generation
X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 24)

fig = plt.figure(0)
plt.grid(True)
plt.scatter(X[:,0],X[:,1])
plt.show()

# standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# randomly initialize cluster centers
k = 3
clusters = {}

np.random.seed(24)

for idx in range(k):
    center = 2*(2*np.random.random((X.shape[1],))-1)
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }
    
    clusters[idx] = cluster
#    print('clusters: ', clusters,'\n')
# plotting the initial cluster centers
plt.scatter(X[:,0],X[:,1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '*',c = 'red')
plt.show()

# define a euclidean distance function
def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

# assign points to the nearest cluster center
def assign_points_to_cluster(X,clusters):
    for i in range(X.shape[0]):
        dist = []
        point = X[i]
        min_distance = float('inf')
        for j in clusters:
            center = clusters[j]['center']
            distance = euclidean_distance(point,center)
            dist.append(distance)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = j
        clusters[closest_cluster]['points'].append(point)
    return clusters
def update_cluster_centers(clusters):
    for i in clusters:
        points = clusters[i]['points']
        if len(points) > 0:
            new_center = np.mean(points,axis = 0)
            clusters[i]['center'] = new_center
    return clusters

def predict(X,clusters):
    y_pred = []
    for i in range(X.shape[0]):
        dist = []
        point = X[i]
        min_distance = float('inf')
        for j in clusters:
            center = clusters[j]['center']
            distance = euclidean_distance(point,center)
            dist.append(distance)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = j
        y_pred.append(closest_cluster)
    return np.array(y_pred)
#assigning points to the nearest cluster center
clusters = assign_points_to_cluster(X,clusters)
clusters = update_cluster_centers(clusters)
pred = predict(X,clusters)

# plotting the clusters
plt.scatter(X[:,0],X[:,1],c = pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[1],marker = '^',c = 'red')
plt.show()