# aadhaar-open-data
We’ll walk through Python basics, data analysis, visualization, and machine learning step by step.  No prior experience is required – perfect for students, beginners, and anyone curious about AI/ML!

The Aadhaar Enrolment and Update Data catalog provides monthly statistics on Aadhaar enrolments and updates across India, categorized by age group and pin code. Released under the National Data Sharing and Accessibility Policy (NDSAP), it is part of the Open Government Data (OGD) Platform India, ensuring transparency and accessibility of nationwide demographic information.

# Register on the site and dowload the data
https://www.data.gov.in/catalog/aadhaar-enrolment-and-update-data

We can do a couple of things with this dataset
Let`s start with KMeans

## K-Means is an unsupervised machine learning algorithm used to group similar data points into clusters.
It doesn’t need labels (like supervised learning). Instead, it only looks at the features and tries to find patterns.

“K” = number of clusters you want.

“Means” = each cluster is represented by the mean (centroid) of its data points.

## How K-Means Works (Detailed + Simple)

Choose K (number of clusters).

Initialize Centroids (randomly select K points from the dataset).

Assign Step: Each data point is assigned to the nearest centroid (Euclidean distance is commonly used).

Update Step: Recalculate the centroid of each cluster (mean of all points in that cluster).

Repeat steps 3 & 4 until centroids stop moving (convergence).

## How to upload the downloaded csv file in google colab
<img width="1118" height="677" alt="image" src="https://github.com/user-attachments/assets/8872ce27-4ed0-4704-9e11-c905feb935eb" />

## Click on 3 dots then click on copy path

<img width="364" height="340" alt="image" src="https://github.com/user-attachments/assets/561ceb8c-41b8-463a-b6c4-2f564365622f" />

```py
import pandas as pd

# Load CSV file
df = pd.read_csv("/content/Demographic_update_data_March-July.csv")

print(df.head())  # show first 5 rows
```
<img width="681" height="277" alt="image" src="https://github.com/user-attachments/assets/82faf187-a82a-4889-a308-acb01395e644" />

## Identify unique entries and counts

```py
df["State"].unique()
```
<img width="665" height="210" alt="image" src="https://github.com/user-attachments/assets/a5f58811-d661-4bfe-8c24-0bb36d10af60" />

```py
```

```py
df["State"].value_counts()
```
<img width="496" height="347" alt="image" src="https://github.com/user-attachments/assets/9c0130e6-48ba-4c6b-a28a-d0439eff77dc" />
```py
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```

## Python does not recognize State names so we need to comnvert string to numbers we can use One-Hot encoding or Label Encoder to do so


```py
from sklearn.preprocessing import LabelEncoder

le_state = LabelEncoder()
le_district = LabelEncoder()

df["State_num"] = le_state.fit_transform(df["State"])
df["District_num"] = le_district.fit_transform(df["District"])

```
## Let`s perform clustering

```py
# 2. Features for clustering
# ---------------------------
X = df[["State_num", "District_num", "Demo_age_5_17", "Demo_age_17+"]]

# ---------------------------
# 3. Scale features
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 4. Apply KMeans
# ---------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ---------------------------
# 5. Check results
# ---------------------------
print(df[["State", "District", "Demo_age_5_17", "Demo_age_17+", "Cluster"]].head())

```

## Vizualizing with plots

```py
# 6. Visualize clusters
# ---------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["Demo_age_5_17"], df["Demo_age_17+"], 
            c=df["Cluster"], cmap="viridis", s=80)
plt.xlabel("Demo_age_5_17")
plt.ylabel("Demo_age_17+")
plt.title("KMeans Clustering (State + District + Age)")
plt.colorbar(label="Cluster")
plt.show()
```
<img width="703" height="547" alt="image" src="https://github.com/user-attachments/assets/00713be9-b42f-43e9-9810-eb684c74737a" />

## But how do I decide the number of cluster - Let`s use Elbow method


```py
# ---------------------------
# 4. Elbow method
# ---------------------------
inertia = []
K = range(1, 11)  # test cluster numbers from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # inertia = sum of squared distances

# ---------------------------
# 5. Plot elbow curve
# ---------------------------
plt.figure(figsize=(8,6))
plt.plot(K, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method for Optimal k")
plt.show()
```
The curve starts to flatten for the first time when k = 4

<img width="713" height="547" alt="image" src="https://github.com/user-attachments/assets/e29b75f6-104e-48ff-888c-aac303cf2e26" />


```py
cluster_summary = df.groupby(["State", "Cluster"]).size().unstack(fill_value=0)
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap: clusters per state
plt.figure(figsize=(12,8))
sns.heatmap(cluster_summary, annot=True, fmt="d", cmap="YlGnBu")
plt.title("District Distribution Across Clusters by State")
plt.show()
```
<img width="1226" height="701" alt="image" src="https://github.com/user-attachments/assets/9fb16368-e3b1-40c1-9db0-457a4af922e5" />

## Example Interpretations

Example Interpretations

✅Andaman and Nicobar Islands

All 8 records went to Cluster 0.

Means this state’s data is homogeneous and fits into one cluster.


✅Delhi

155 in Cluster 0, 152 in Cluster 1, 20 in Cluster 3.

Unlike most states, Delhi’s records are spread across multiple clusters, suggesting greater internal diversity in the data (districts vary more in demographics).




