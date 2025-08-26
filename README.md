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
