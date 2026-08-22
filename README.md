# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Mall Customers dataset and select Annual Income and Spending Score as the input features.
2.Initialize the K-Means algorithm with 5 clusters and assign each customer to the nearest cluster based on the selected features.
3.Calculate and update the cluster centroids repeatedly until the clusters become stable.
4.Plot the customer clusters and their centroids using a scatter plot to visualize the customer segmentation.


## Program:
```py
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MUKHIL S
RegisterNumber:  212225040263
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
X = data.iloc[:, [3, 4]].values
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=100)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
```

## Output:
<img width="536" height="382" alt="image" src="https://github.com/user-attachments/assets/1f48cfa4-09c9-4469-ba66-ea2eb6598d20" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
