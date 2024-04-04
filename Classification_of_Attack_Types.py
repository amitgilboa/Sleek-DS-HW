#### Run this file after running Label_kmeans.py file

# Import libraries
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Split the data to train and test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_new, test_size=0.2, random_state=42, stratify=y_new)

# Normlaize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Run XGBoost
xgb_classifier = XGBClassifier(objective='multi:softmax', num_class=len(np.unique(y_new)))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

precision_scores_xg = []
recall_scores_xg = []
xg_top10_features_all = []

for fold, (train_index, val_index) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"Fold {fold + 1}:")
    # Split the data into train and validation sets
    X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    # Fit the model on the training data for this fold
    xgb_classifier.fit(X_train_fold, y_train_fold)
    ## feature importance
    xgb_feature_importance = xgb_classifier.feature_importances_
    top10_features = X_train.columns[np.argsort(xgb_classifier.feature_importances_)[::-1][:10]]
    xg_top10_features_all.append(top10_features)
    # Predict labels on the validation data for this fold
    y_val_pred = xgb_classifier.predict(X_val_fold)
    # Calculate precision and recall for this fold
    precision = precision_score(y_val_fold, y_val_pred, average='weighted')
    recall = recall_score(y_val_fold, y_val_pred, average='weighted')
    # Print precision and recall for this fold
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    precision_scores_xg.append(precision)
    recall_scores_xg.append(recall)

np.mean(precision_scores_xg)
np.mean(recall_scores_xg)
xg_top10_features_all = np.unique(np.concatenate(xg_top10_features_all))

# After training, you can fit the model on the entire training data
xgb_classifier.fit(X_train_scaled, y_train)
y_test_pred = xgb_classifier.predict(X_test_scaled)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
