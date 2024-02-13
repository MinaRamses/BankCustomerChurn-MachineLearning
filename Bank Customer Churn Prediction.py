# Bank Customer Churn Prediction

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, classification_report, mean_squared_error, r2_score, mean_absolute_error, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, svm, ensemble
# Import Data
df = pd.read_csv("Churn_Modelling.csv",sep = ',')
df_ml = df
## Show Data
df
df_ml
df.describe()
df.info()
df.shape
df["Exited"].unique()
df["Geography"].unique()
df["Gender"].unique()
df.dtypes
### Show all columns names
df.columns
# Pre-processing steps 
## Data Cleaning
## Check for NULL values
df.isna().sum()
## Check for DUPLICATED values
df.duplicated().sum()
first_rows = df.head()


for column in first_rows.columns:
    column_values = first_rows [column]
    value = column_values. values[0]
    value_type = type(value).__name__
    print(f"Column: {column} - Value: {value} - Type: {value_type} \n")
# Removing irrelevant columns such as name and customerid
df.drop(['RowNumber','CustomerId','Surname'],inplace=True,axis=1)
# Mode - Median - Mean for Exited 
mode = df['Age'][df['Exited'] == 0].mode()[0]
mean = df['Age'][df['Exited'] == 0].mean()
median = df['Age'][df['Exited'] == 0].median()
mode_exit = df['Age'][df['Exited'] == 1].mode()[0]
mean_exit= df['Age'][df['Exited'] == 1].mean()
median_exit = df['Age'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")
# Data Transformation
## Encoding Categorical Variables
# Before
df["Gender"].unique()
gender_counts = df['Gender'].value_counts()
print("Number of Male Customers:", gender_counts['Male'])
print("Number of Female Customers:", gender_counts['Female'])
# Perform binary encoding for 'Gender'
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# After
df["Gender"].unique()
## Feature Engineering
# Create a new feature 'TenureAgeRatio'
df['TenureAgeRatio'] = df['Tenure'] / df['Age']
print(df.head())
df.info()
# Display unique values for the 'Exited', 'Geography', and 'Gender' columns
print(df["Exited"].unique())
print(df["Geography"].unique())
print(df["Gender"].unique())
# Graphs & Analysis
# Distribution of customer demographics
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=30, kde=True, color='navy', alpha=0.7)
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Perform binary encoding for 'Gender'
df['Gender'] = df['Gender'].map({0: 'Male', 1 : 'Female'})

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender')
plt.title('Distribution of Customer Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# Perform binary encoding for 'Gender'
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})


plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Geography')
plt.title('Distribution of Customer Geography')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()


### Calculate the number of customers in each country
country_counts = df['Geography'].value_counts()

print("Number of Customers in Each Country:")
print(country_counts)
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Gender', bins=30, kde=True, alpha=0.7)
plt.title('Customer Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend(title='Gender', labels=['Male', 'Female'])  # Specify labels for the legend
plt

plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='NumOfProducts', y='Exited', errorbar=None, palette='coolwarm')
plt.title('Churn Rate by Product Type')
plt.xlabel('Number of Products')
plt.ylabel('Churn Rate')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Tenure', bins=30, kde=True, color='purple', alpha=0.7, label='Tenure Distribution')
plt.title('Customer Tenure Distribution')
plt.xlabel('Tenure (Years)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

X = df[['Age', 'Balance']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Balance', hue='Cluster', palette='Set1')
plt.title('Customer Segmentation Analysis')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.legend(title='Cluster')
plt.show()
# Trends in customer account balances and tenure
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Tenure', y='Balance', hue='Exited', palette='coolwarm')
plt.title('Trends in Customer Account Balances and Tenure')
plt.xlabel('Tenure (Years)')
plt.ylabel('Account Balance')
plt.show()

# Correlations between different features
plt.figure(figsize=(10, 8))
corr = df[['Age', 'Balance', 'NumOfProducts', 'Exited']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)

q1_exit_0 = df['Age'][df['Exited'] == 0].quantile(0.25)
q3_exit_0 = df['Age'][df['Exited'] == 0].quantile(0.75)

q1_exit_1 = df['Age'][df['Exited'] == 1].quantile(0.25)
q3_exit_1 = df['Age'][df['Exited'] == 1].quantile(0.75)


print("Quartiles for Age Distribution when Exited = 0:")
print(f"Q1: {q1_exit_0}, Q3: {q3_exit_0}\n")

print("Quartiles for Age Distribution when Exited = 1:")
print(f"Q1: {q1_exit_1}, Q3: {q3_exit_1}")
## Bar Chart for all columns except (Balance, Estimated Salary and CreditScore) VS Exited 
for i , c in enumerate(df.drop(columns=['Exited','Balance','EstimatedSalary','CreditScore'])):
    plt.figure(i, figsize=(15, 5))
    sns.countplot(data=df,x=c,hue='Exited')
# Mode, Mean and Median
mode = df['CreditScore'][df['Exited'] == 0].mode()[0]
mean = df['CreditScore'][df['Exited'] == 0].mean()
median = df['CreditScore'][df['Exited'] == 0].median()
mode_exit = df['CreditScore'][df['Exited'] == 1].mode()[0]
mean_exit= df['CreditScore'][df['Exited'] == 1].mean()
median_exit = df['CreditScore'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")
mode = df['EstimatedSalary'][df['Exited'] == 0].mode()[0]
mean = df['EstimatedSalary'][df['Exited'] == 0].mean()
median = df['EstimatedSalary'][df['Exited'] == 0].median()
mode_exit = df['EstimatedSalary'][df['Exited'] == 1].mode()[0]
mean_exit= df['EstimatedSalary'][df['Exited'] == 1].mean()
median_exit = df['EstimatedSalary'][df['Exited'] == 1].median()

print("-----------------------------------------------------")
print("|    Statistics    |  Exited = 0 |  Exited= 1       |")
print("-----------------------------------------------------")
print(f"| Mode             |  {mode:<9}  |  {mode_exit:<14}  |")
print(f"| Median           |  {median:<9}  |  {median_exit:<14.2f}  |")
print(f"| Mean             |  {mean:<9.2f}  |  {mean_exit:<14.2f}  |")
print("-----------------------------------------------------")
## Estimated Salary Distribution for Exited and Not Exited
plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
sns.histplot(data=df.loc[df['Exited'] == 0], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Not Exited")

plt.subplot(1, 2, 2)
sns.histplot(data=df.loc[df['Exited'] == 1], x="EstimatedSalary", kde=True, color=sns.color_palette("viridis")[1])
plt.title("Estimated Salary Distribution for Exited")

plt.tight_layout()
plt.show()
# This is an error text to stop the code before doing ML

# STOP CODE
# Churn Count
churn_count = df[df['Exited'] == 1].groupby('Geography').size().reset_index(name='churn_count')

non_churn_count = df[df['Exited'] == 0].groupby('Geography').size().reset_index(name='non_churn_count')

combined_count = churn_count.merge(non_churn_count, on='Geography')

total_count = df['Geography'].value_counts().reset_index()
total_count.columns = ['Geography', 'total_count']
combined_count = combined_count.merge(total_count, on='Geography')
combined_count['churn_percentage'] = (combined_count['churn_count'] / combined_count['total_count']).round(4) * 100

combined_count
df_ml.head()
df_ml['Geography'] = df_ml['Geography'].map({'France': 1, 'Germany': 2, 'Spain': 3})
print(df.columns)

# Calculate correlation matrix
correlation_matrix = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'TenureAgeRatio']].corr()
print(correlation_matrix)

# Checking the Relationship of Each Two Important Features
important_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited', 'TenureAgeRatio']
sns.pairplot(df[important_features], hue='Exited', diag_kind='kde')
plt.show()

plt.figure(figsize=(15,8))
df.corr()['Exited'].sort_values(ascending =False).plot(kind='bar',title='Correlation Graph vs Exited')
plt.grid()
## Calculate the correlation of each feature with 'Exited' and sort in descending order
correlation_with_exited = df.corr()['Exited'].sort_values(ascending=False)
print("Correlation of each feature with 'Exited':")
print(correlation_with_exited)

# MACHINE LEARNING
## Trainning different ML technique
# Random Forest
X = df_ml.drop(columns=['Exited']) 
y = df_ml['Exited'] 

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate a confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Decision Tree
# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier()

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy:", accuracy_dt)

# Generate a classification report
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Generate a confusion matrix
print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# XGBoost
# Initialize the XGBoost classifier
xgb_classifier = XGBClassifier()

# Train the classifier on the training data
xgb_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_xgb = xgb_classifier.predict(X_test)

# Evaluate the performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", accuracy_xgb)

# Generate a classification report
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Generate a confusion matrix
print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Calculating TP, TN, FP, FN, F1, Recall, Precision
## Random Forest
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_classifier.predict(X_test)

# Calculate the confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

# Calculate Precision, Recall, and F1-score
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Display the results
print("Random Forest Metrics:")
print("True Positives:", tp_rf)
print("True Negatives:", tn_rf)
print("False Positives:", fp_rf)
print("False Negatives:", fn_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)

## Decision Tree Classifier
# Calculate the confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

# Calculate Precision, Recall, and F1-score
precision_dt = precision_score(y_test, y_pred_dt)
recall_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# Display the results
print("Decision Tree Metrics:")
print("True Positives:", tp_dt)
print("True Negatives:", tn_dt)
print("False Positives:", fp_dt)
print("False Negatives:", fn_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-score:", f1_dt)

## XGBoost Classifier
# Calculate the confusion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()

# Calculate Precision, Recall, and F1-score
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

# Display the results
print("XGBoost Metrics:")
print("True Positives:", tp_xgb)
print("True Negatives:", tn_xgb)
print("False Positives:", fp_xgb)
print("False Negatives:", fn_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1_xgb)

# ROC Plot and Score
### Plot all 3 MLs ROC AUC in a single graph
# Calculate ROC curve and AUC for Random Forest
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_classifier.predict_proba(X_test)[:,1])
auc_rf = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:,1])

# Plot ROC curve for Random Forest
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')

# Calculate ROC curve and AUC for Decision Tree
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, dt_classifier.predict_proba(X_test)[:,1])
auc_dt = roc_auc_score(y_test, dt_classifier.predict_proba(X_test)[:,1])

# Plot ROC curve for Decision Tree
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')

# Calculate ROC curve and AUC for XGBoost
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, xgb_classifier.predict_proba(X_test)[:,1])
auc_xgb = roc_auc_score(y_test, xgb_classifier.predict_proba(X_test)[:,1])

# Plot ROC curve for XGBoost
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})')

# Plot ROC curve for random guessing (AUC = 0.5)
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guessing (AUC = 0.5)', color='gray')

# Add labels and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

## Print out the ROC AUC scores
print("ROC AUC for Random Forest:", auc_rf)
print("ROC AUC for Decision Tree:", auc_dt)
print("ROC AUC for XGBoost:", auc_xgb)
# Comparison Between Different Machine Learning with Graphs
## Calculate metrics for each model
models = ['Random Forest', 'Decision Tree', 'XGBoost']
accuracy = [accuracy_score(y_test, model.predict(X_test)) for model in [rf_classifier, dt_classifier, xgb_classifier]]
precision = [precision_score(y_test, model.predict(X_test)) for model in [rf_classifier, dt_classifier, xgb_classifier]]
recall = [recall_score(y_test, model.predict(X_test)) for model in [rf_classifier, dt_classifier, xgb_classifier]]
f1 = [f1_score(y_test, model.predict(X_test)) for model in [rf_classifier, dt_classifier, xgb_classifier]]
## Plot bar plots for accuracy, precision, recall, and F1-score
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.bar(models, accuracy, color='skyblue')
plt.title('Accuracy')
plt.ylabel('Score')

plt.subplot(2, 2, 2)
plt.bar(models, precision, color='salmon')
plt.title('Precision')
plt.ylabel('Score')

plt.subplot(2, 2, 3)
plt.bar(models, recall, color='lightgreen')
plt.title('Recall')
plt.ylabel('Score')

plt.subplot(2, 2, 4)
plt.bar(models, f1, color='gold')
plt.title('F1-Score')
plt.ylabel('Score')

plt.tight_layout()
plt.show()
## Metrics Data
### Entered the data manually
models = ['Random Forest', 'Decision Tree', 'XGBoost']
accuracy = [0.8685, 0.79, 0.8585]
precision = [0.7763, 0.4691, 0.6884]
recall = [0.4504, 0.5216, 0.5115]
f1 = [0.5700, 0.4940, 0.5869]

# Plotting
plt.figure(figsize=(10, 6))

# Accuracy
plt.bar([i-0.2 for i in range(len(models))], accuracy, color='skyblue', width=0.2, label='Accuracy')
# Precision
plt.bar([i for i in range(len(models))], precision, color='salmon', width=0.2, label='Precision')
# Recall
plt.bar([i+0.2 for i in range(len(models))], recall,color= 'lightgreen', width=0.2, label='Recall')
# F1-score
plt.bar([i+0.4 for i in range(len(models))], f1, color='gold', width=0.2, label='F1-score')

plt.xlabel('Models')
plt.ylabel('Metrics')
plt.title('Performance Metrics Comparison')
plt.xticks(range(len(models)), models)
plt.legend()
plt.grid(True)
plt.show()

## Plot confusion matrices for each model
plt.figure(figsize=(15, 5))
for i, model in enumerate([rf_classifier, dt_classifier, xgb_classifier], 1):
    plt.subplot(1, 3, i)
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {models[i-1]}')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, ['Not Exited', 'Exited'])
    plt.yticks(tick_marks, ['Not Exited', 'Exited'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plt.tight_layout()
plt.show()
## Plot precision-recall curves for each model
plt.figure(figsize=(8, 6))
for model, name in zip([rf_classifier, dt_classifier, xgb_classifier], models):
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(recall, precision, label=name)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
## Plot F1-score curves for each model
plt.figure(figsize=(8, 6))
for model, name in zip([rf_classifier, dt_classifier, xgb_classifier], models):
    precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    f1 = 2 * (precision * recall) / (precision + recall)
    plt.plot(thresholds, f1[:-1], label=name)

plt.xlabel('Threshold')
plt.ylabel('F1-Score')
plt.title('F1-Score Curve')
plt.legend()
plt.grid(True)
plt.show()

# Plot F1-score curves for each model
plt.figure(figsize=(8, 6))
for model, name in zip([rf_classifier, dt_classifier, xgb_classifier], models):
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    f1 = 2 * (precision * recall) / (precision + recall)
    plt.plot(recall[:-1], f1[:-1], label=name)  # Use recall[:-1] instead of recall

plt.xlabel('Recall')
plt.ylabel('F1-Score')
plt.title('F1-Score Curve')
plt.legend()
plt.grid(True)
plt.show()

