# Import libraries
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

merged_df2 = pd.read_pickle("merged_df2.pkl")

# I remove BENIGN labels to classify differnet types of attacks
merged_df2_filter = merged_df2[merged_df2['Label']!='BENIGN']

numeric_data = merged_df2_filter.select_dtypes(include=[np.number])
numeric_data.drop(columns=['Source Port','Destination Port'], inplace=True) # remove this 2 columns which are categorical and just have values as categorics

numeric_data['Label'] = merged_df2_filter['Label']
numeric_data['Label'].value_counts()

# Remove null values in each column
numeric_data.fillna(0, inplace=True)

# Check which columns have many 0 values:
zero_percentages = (numeric_data == 0).mean() * 100
zero_percentages_sorted = zero_percentages.sort_values(ascending=False)
remove_columns = zero_percentages_sorted[zero_percentages_sorted>50].index.tolist()
numeric_data = numeric_data.drop(columns=remove_columns, inplace=False)

numeric_data = numeric_data[numeric_data['Label']!=0]
X = numeric_data.drop(['Label'], axis=1)  # Features
y = numeric_data['Label']  # Target

# Check for infinite values in the DataFrame
inf_values_mask = np.isinf(X)
# Count the number of infinite values
num_infinite_values = np.count_nonzero(inf_values_mask)
num_infinite_values
# remove those rows
X.replace([np.inf, -np.inf], np.nan, inplace=True)
dropped_indices = X.index.difference(X.dropna().index)

X = X.drop(dropped_indices, axis=0)
y = y.drop(dropped_indices)

# label y :
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

### Find outliers
from scipy import stats
z_scores = stats.zscore(X) # Calculate Z-scores for each numerical column
threshold = 3 # Define a threshold for identifying outliers (e.g., Z-score greater than 3)
outliers = np.where(np.abs(z_scores) > threshold) # Find outliers based on Z-scores
len(outliers[0]) # There are many outliers so we use StandardScaler

outliers_count = (np.abs(z_scores) > threshold).sum(axis=0)
total_observations = X.shape[0]
outliers_frequency_pct = (outliers_count / total_observations) * 100
print("Frequency of outliers in each column (in %):")
print(outliers_frequency_pct.sort_values())

columns_to_keep = outliers_frequency_pct < 3 # I choose a threshold of 3 but we can change it to improve the classification results
X_filtered = X.loc[:, columns_to_keep]

# Apply K-means to have bigger attack label categories 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Encode the categorical labels into numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Define the number of clusters
num_clusters = 4
# Initialize the KMeans model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(y_encoded.reshape(-1, 1)) # Fit the KMeans model to the encoded labels
cluster_centers = kmeans.cluster_centers_ # Get the cluster centers
cluster_labels = kmeans.predict(y_encoded.reshape(-1, 1)) # Assign each data point to the nearest cluster

# Create a dictionary
df = pd.DataFrame(np.column_stack([y, cluster_labels]), columns=['cluster_labels_original', 'cluster_labels'])
cluster_map = df.groupby('cluster_labels_original')['cluster_labels'].unique().apply(list).to_dict()

print("Cluster Label Map:")
for original_label, assigned_labels in cluster_map.items():
    print(f"Original Label: {original_label}, Assigned Labels: {assigned_labels}")

# Define a new variable of our clusters
y_new = y.map(cluster_map).apply(lambda x: x[0])

np.unique(y_new, return_counts=True)
# The results are: (array([0, 1, 2, 3], dtype=int64), array([166917, 245915, 129981,  13742], dtype=int64))

# We will use the y_new as the target variable in the Attack classification model


