import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")

# KMeans model
model = KMeans(n_clusters=2)
model.fit(data)

# Get clusters
labels = model.labels_

# Plot clusters
plt.scatter(data['Age'], data['Spending'], c=labels)
plt.xlabel("Age")
plt.ylabel("Spending")
plt.title("Customer Segmentation using K-Means")
plt.show()
