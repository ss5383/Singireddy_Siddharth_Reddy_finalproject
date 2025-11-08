#!/usr/bin/env python
# coding: utf-8

# Data Mining - Final Term Project(ss5383)

# In[1]:


# TensorFlow Installation
#pip install tensorflow==2.18.0


# In[2]:


get_ipython().system('pip show tensorflow')


# In[3]:


# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, brier_score_loss, 
    accuracy_score, roc_curve, auc
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

warnings.filterwarnings("ignore")



# In[4]:


# Load Dataset from GitHub
dataset_url = "https://github.com/ss5383/Singireddy_Siddharth_Reddy_finalproject/blob/main/Heart_Disease_Dataset.csv"
try:
    df = pd.read_csv(dataset_url)
    if df.empty:
        print("Dataset is empty.")
    else:
        print(f"Dataset successfully loaded with shape: {df.shape}")
except FileNotFoundError:
    print("File not found. Verify the URL or path.")
except pd.errors.EmptyDataError:
    print("File exists but contains no data.")
except Exception as error:
    print(f"An error occurred: {error}")

print("\nDataset Information:")
print(df.info())

# Check for Null Values
print("\nMissing Values Check:")
print(df.isnull().sum())

# Preview the Dataset
print("\nInitial Dataset Snapshot:")
print(df.head())



# In[5]:


# Identify and Encode Categorical Columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("\nUnique values in Categorical Columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")

encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nDataset after Label Encoding:")
print(df.head())



# In[6]:


# Feature and Label Separation
X = df.iloc[:, :-1]
y = df.iloc[:, -1]



# In[7]:


# Target Distribution Visualization
sns.countplot(x=y, palette=["green", "red"])
plt.xlabel("Heart Disease (0: No, 1: Yes)")
plt.ylabel("Frequency")
plt.title("Target Class Distribution")
plt.show()



# In[8]:


# Display Distribution Percentages
pos, neg = y.value_counts()
total = y.count()
print(f"\n{(neg/total)*100:.2f}% instances are 'No Heart Disease' ({neg})")
print(f"{(pos/total)*100:.2f}% instances are 'Yes Heart Disease' ({pos})")



# In[9]:


# Correlation Matrix
plt.figure(figsize=(10, 8))
correlation = X.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()



# In[10]:


# Histograms
X.hist(figsize=(10, 10))
plt.suptitle("Histograms for Each Feature", y=1.02)
plt.tight_layout()
plt.show()



# In[11]:


# Pairwise Relationships
sns.pairplot(df, hue="HeartDisease")
plt.suptitle("Pairplot of All Features", y=1.02)
plt.show()



# In[12]:


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# In[13]:


# Data Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# In[14]:


# Reshape for LSTM Input
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# In[15]:


# Utility: Confusion Matrix Plotting
def visualize_conf_matrix(matrix, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    plt.show()


# In[16]:


# Utility: ROC-AUC Curve Plotting
def draw_roc_auc(model_name, fpr, tpr, auc_score):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="blue")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC AUC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


# In[17]:


# Metrics Calculator
def evaluate_model(name, y_train_fold, y_test_fold, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred).ravel()

    print(f"\nConfusion Matrix ({name}):\n", confusion_matrix(y_test_fold, y_pred))
    visualize_conf_matrix(confusion_matrix(y_test_fold, y_pred), name)

    try:
        fpr, tpr, _ = roc_curve(y_test_fold, y_prob)
        auc_val = auc(fpr, tpr)
        draw_roc_auc(name, fpr, tpr, auc_val)
    except:
        print("ROC curve could not be plotted due to insufficient class variance.")

    # Calculate Metrics
    metrics = {
        "True Positive (TP)": tp,
        "True Negative (TN)": tn,
        "False Positive (FP)": fp,
        "False Negative (FN)": fn,
        "Sensitivity (TPR)": tp / (tp + fn) if (tp + fn) else 0,
        "Specificity (TNR)": tn / (tn + fp) if (tn + fp) else 0,
        "False Positive Rate (FPR)": fp / (fp + tn) if (fp + tn) else 0,
        "False Negative Rate (FNR)": fn / (fn + tp) if (fn + tp) else 0,
        "Recall (r)": tp / (tp + fn) if (tp + fn) else 0,
        "Precision (P)": tp / (tp + fp) if (tp + fp) else 0,
        "F1 Measure (F1)": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Error Rate": (fp + fn) / (tp + tn + fp + fn),
        "Balanced Accuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if (tp + fn and tn + fp) else 0,
        "True Skill Statistics (TSS)": (tp / (tp + fn)) - (fp / (fp + tn)) if (tp + fn and fp + tn) else 0,
        "Heidke Skill Score (HSS)": (2 * ((tp * tn) - (fp * fn))) / (((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))) if (((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn)) > 0) else 0,
        "ROC_AUC Score": roc_auc_score(y_test_fold, y_prob),
        "Brier Score": brier_score_loss(y_test_fold, y_prob)
    }

    # Brier Skill Score
    baseline_prob = [y_train_fold.mean()] * len(y_test_fold)
    baseline_brier = brier_score_loss(y_test_fold, baseline_prob)
    metrics["Brier Skill Score"] = 1 - (metrics["Brier Score"] / baseline_brier) if baseline_brier else 0

    return metrics


# In[18]:


# Random Forest Model Function
def run_random_forest(name, results, X_train_fold, y_train_fold, X_test_fold, y_test_fold):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_fold, y_train_fold)
    predictions = model.predict(X_test_fold)
    prob_scores = model.predict_proba(X_test_fold)[:, 1]
    results.append(evaluate_model(name, y_train_fold, y_test_fold, predictions, prob_scores))
    return results


# In[19]:


# K-Nearest Neighbors Function
def run_knn(name, results, X_train_fold, y_train_fold, X_test_fold, y_test_fold):
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train_fold, y_train_fold)
    predictions = model.predict(X_test_fold)
    prob_scores = model.predict_proba(X_test_fold)[:, 1]
    results.append(evaluate_model(name, y_train_fold, y_test_fold, predictions, prob_scores))
    return results


# In[20]:


# LSTM Model Function
def run_lstm(name, results, X_train_lstm, y_train_fold, X_test_lstm, y_test_fold):
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_lstm, y_train_fold, epochs=10, batch_size=16, verbose=0)

    predictions = (model.predict(X_test_lstm) > 0.5).astype("int32")
    prob_scores = model.predict(X_test_lstm).flatten()
    results.append(evaluate_model(name, y_train_fold, y_test_fold, predictions, prob_scores))
    return results


# In[21]:


# Lists to hold metrics for each fold
rf_train_metrics, knn_train_metrics, lstm_train_metrics = [], [], []
fold_labels = []


# In[22]:


# Initialize Stratified K-Fold for imbalanced dataset
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for idx, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\nRunning Fold {idx}")
    fold_labels.append(f"Fold_{idx}")

    # Fold-specific splits
    X_train_fold = X_train.iloc[train_idx]
    X_test_fold = X_train.iloc[test_idx]
    y_train_fold = y_train.iloc[train_idx]
    y_test_fold = y_train.iloc[test_idx]

    # Scaling features
    fold_scaler = StandardScaler()
    X_train_fold_scaled = fold_scaler.fit_transform(X_train_fold)
    X_test_fold_scaled = fold_scaler.transform(X_test_fold)

    # LSTM reshaping
    X_train_fold_lstm = X_train_fold_scaled.reshape(-1, X_train_fold_scaled.shape[1], 1)
    X_test_fold_lstm = X_test_fold_scaled.reshape(-1, X_test_fold_scaled.shape[1], 1)

    # Train and Evaluate Models
    run_random_forest("Random_Forest", rf_train_metrics, X_train_fold_scaled, y_train_fold, X_test_fold_scaled, y_test_fold)
    run_knn("K_Nearest_Neighbor", knn_train_metrics, X_train_fold_scaled, y_train_fold, X_test_fold_scaled, y_test_fold)
    run_lstm("LSTM", lstm_train_metrics, X_train_fold_lstm, y_train_fold, X_test_fold_lstm, y_test_fold)


# In[23]:


# Convert training fold results to DataFrames
rf_df = pd.DataFrame(rf_train_metrics).T
rf_df.columns = fold_labels

knn_df = pd.DataFrame(knn_train_metrics).T
knn_df.columns = fold_labels

lstm_df = pd.DataFrame(lstm_train_metrics).T
lstm_df.columns = fold_labels


# In[24]:


# Display per-fold metrics
print("\nRandom Forest - KFold Training Metrics:")
print(rf_df)
print("\nK-Nearest Neighbors - KFold Training Metrics:")
print(knn_df)
print("\nLSTM - KFold Training Metrics:")
print(lstm_df)


# In[25]:


# Compute Average Performance per Algorithm
rf_avg = rf_df.mean(axis=1)
knn_avg = knn_df.mean(axis=1)
lstm_avg = lstm_df.mean(axis=1)

average_summary = pd.DataFrame({
    "Random Forest": rf_avg,
    "K-Nearest Neighbor": knn_avg,
    "LSTM": lstm_avg
})

print("\nAverage Cross-Validation Performance Metrics:")
print(average_summary)


# In[26]:


# Scale the full training and test sets
final_scaler = StandardScaler()
X_train_final = final_scaler.fit_transform(X_train)
X_test_final = final_scaler.transform(X_test)


# In[27]:


# Reshape for LSTM input
X_train_final_lstm = X_train_final.reshape(-1, X_train_final.shape[1], 1)
X_test_final_lstm = X_test_final.reshape(-1, X_test_final.shape[1], 1)


# In[28]:


# Lists to store final test results
rf_test_metrics, knn_test_metrics, lstm_test_metrics = [], [], []


# In[29]:


# Final model evaluations on unseen test set
run_random_forest("Random_Forest_Test", rf_test_metrics, X_train_final, y_train, X_test_final, y_test)
run_knn("K_Nearest_Neighbor_Test", knn_test_metrics, X_train_final, y_train, X_test_final, y_test)
run_lstm("LSTM_Test", lstm_test_metrics, X_train_final_lstm, y_train, X_test_final_lstm, y_test)


# In[30]:


# Convert to DataFrames
rf_test_df = pd.DataFrame(rf_test_metrics).T
knn_test_df = pd.DataFrame(knn_test_metrics).T
lstm_test_df = pd.DataFrame(lstm_test_metrics).T


# In[31]:


# Combine Test Metrics
test_results_df = pd.concat([rf_test_df, knn_test_df, lstm_test_df], axis=1)
test_results_df.columns = ["Random Forest", "K-Nearest Neighbor", "LSTM"]


# In[32]:


print("\nTest Set Performance Comparison:")
print(test_results_df)


# In[33]:


# Identify Best Performing Algorithm on Test Accuracy
test_accuracy_scores = {
    "Random Forest": rf_test_metrics[0]["Accuracy"],
    "K-Nearest Neighbor": knn_test_metrics[0]["Accuracy"],
    "LSTM": lstm_test_metrics[0]["Accuracy"]
}
best_test_model = max(test_accuracy_scores, key=test_accuracy_scores.get)
print(f"\nBest Accuracy on Test Set: {best_test_model} ({test_accuracy_scores[best_test_model] * 100:.2f}%)")


# In[34]:


# Best Train Accuracy (based on average CV accuracy)
train_accuracy_scores = {
    "Random Forest": rf_avg["Accuracy"],
    "K-Nearest Neighbor": knn_avg["Accuracy"],
    "LSTM": lstm_avg["Accuracy"]
}
best_train_model = max(train_accuracy_scores, key=train_accuracy_scores.get)
print(f"Best Accuracy on Train Set: {best_train_model} ({train_accuracy_scores[best_train_model] * 100:.2f}%)")


# In[ ]:




