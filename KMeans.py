# Clustering mit Hilfe der KMeans Methode.
# Zusätzlich die Elbow-Method zur Bestimmung der optimalen Anzahl an Clustern.

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("./autos_prepared.csv")
print(df.head())

# Daten visualisieren
plt.scatter(df["yearOfRegistration"], df["price"])
plt.show()


X = df[["yearOfRegistration", "price"]]

scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Elbow-Method zur Bestimmung der besten Anzahl an Clustern
scores = []
for n in range(2, 10):
    model = KMeans(n_clusters = n)
    model.fit(X_transformed)
    scores.append(model.inertia_)

plt.plot(range(2, 10), scores)
plt.show()

# KMeans mit 3 Clustern
model = KMeans(n_clusters = 3)
model.fit(X_transformed)

labels = model.labels_

# Grafische Darstellung
plt.scatter(df["yearOfRegistration"], df["price"], c = labels)
plt.show()










