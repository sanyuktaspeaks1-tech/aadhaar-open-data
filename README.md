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
<img width="496" height="347" alt="image" src="https://github.com/user-attachments/assets/9c0130e6-48ba-4c6b-a28a-d0439eff77dc" />

```


