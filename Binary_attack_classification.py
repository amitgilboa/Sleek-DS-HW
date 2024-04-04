# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

merged_df2 = pd.read_pickle("merged_df2.pkl")

# We saw that the data is imbalanced therefore I will Subset the data to Friday afternoon and Wednesday none because most of the attacks happened there
merged_df2_filter = merged_df2[((merged_df2['Day'] == 'Friday') & (merged_df2['Time'] == 'Afternoon')) | ((merged_df2['Day'] == 'Wednesday') & (merged_df2['Time'] == 'None'))]
merged_df2_filter.shape

## Classify the Label by all the numeric parameters:

numeric_data = merged_df2_filter.select_dtypes(include=[np.number])
numeric_data.drop(columns=['Source Port','Destination Port'], inplace=True) # remove this 2 columns which are categorical and just have values as categorics

numeric_data['Label'] = merged_df2_filter['Label']
numeric_data['Label'].value_counts()

numeric_data['Attack'] = np.where(numeric_data['Label'] == 'BENIGN', 'yes', 'no')
numeric_data['Attack'].value_counts()

# Delete null rows
numeric_data.fillna(0, inplace=True)

# Check which columns have many 0 values:
zero_percentages = (numeric_data == 0).mean() * 100
zero_percentages_sorted = zero_percentages.sort_values(ascending=False)
remove_columns = zero_percentages_sorted[zero_percentages_sorted>50].index.tolist()
numeric_data = numeric_data.drop(columns=remove_columns, inplace=False)

# The data is unbalanced so we will use stratified 5-fold cv
X = numeric_data.drop(['Attack', 'Label'], axis=1)  # Features
y = numeric_data['Attack']  # Target

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

# Encoded y 
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Find outliers
z_scores = stats.zscore(X) # Calculate Z-scores for each numerical column
threshold = 3 # Define a threshold for identifying outliers (e.g., Z-score greater than 3)
outliers = np.where(np.abs(z_scores) > threshold) # Find outliers based on Z-scores
len(outliers[0]) # There are many outliers so we use StandardScaler

outliers_count = (np.abs(z_scores) > threshold).sum(axis=0)
total_observations = X.shape[0]
outliers_frequency_pct = (outliers_count / total_observations) * 100
print("Frequency of outliers in each column (in %):")
print(outliers_frequency_pct.sort_values())

columns_to_keep = outliers_frequency_pct < 3 
X_filtered = X.loc[:, columns_to_keep]

# Split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y)

# Normlaize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply desicion tree
dt_classifier = DecisionTreeClassifier()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

precision_scores = []
recall_scores = []
top10_features_all = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold + 1}:")
    # Split the data into train and validation sets
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    # Fit the model on the training data for this fold
    dt_classifier.fit(X_train_fold, y_train_fold)
    ## feature importance
    dt_feature_importance = dt_classifier.feature_importances_
    top10_features = X_train.columns[np.argsort(dt_classifier.feature_importances_)[::-1][:10]]
    top10_features_all.append(top10_features)
    # Predict labels on the validation data for this fold
    y_val_pred = dt_classifier.predict(X_val_fold)
    # Calculate precision and recall for this fold
    precision = precision_score(y_val_fold, y_val_pred)
    recall = recall_score(y_val_fold, y_val_pred)
    # Print precision and recall for this fold
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    precision_scores.append(precision)
    recall_scores.append(recall)

np.mean(precision_scores)
np.mean(recall_scores)
top10_features_all = np.unique(np.concatenate(top10_features_all))
print(top10_features_all)

# After training, you can fit the model on the entire training data
dt_classifier.fit(X_train_scaled, y_train)
y_test_pred = dt_classifier.predict(X_test_scaled)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

# Compute confusion matrix for test predictions
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

