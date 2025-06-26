# CANCER STAGE PREDICTION



# Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import Dataset

df=pd.read_csv('/content/DNA_Dataset_Normalized.csv')
df.head()

# Data Exploration

df.info()

df.describe().T

# Check For Null Values

# Get the count of null values for each column
null_values=df.isnull().sum()

# Filter to show only columns with one or more null values
cols_with_null_values=null_values[null_values>0]

print("Columns with null values and their counts:")
print(cols_with_null_values)

# Filling null values with the mean of the respective columns
df.fillna(df.mean(),inplace=True)

# Verify that there are no more null values
print(df.isnull().sum().sum())

# Droping column 'gene_23' since it has constant values (zeros)
df.drop('gene_23',axis=1,inplace=True)

# Unique values in data features
for i in df.columns:
  print(i,':',df[i].nunique())

# Data Visualization

# Pairplot of a selected few genes and class
selected=['gene_1','gene_3','gene_4','gene_12','gene_19','gene_30','gene_45','Class']
sns.pairplot(df[selected], diag_kind='kde')
plt.show()

# Countplot of each of the 5 classes
sns.countplot(df,x='Class')

# Piechart
df['Class'].value_counts().plot.pie(autopct="%.0f%%",shadow=True)
plt.show()

From the above two graphs, it is evident that the dataset is balanced, as each class contains an equal number of samples. Therefore, oversampling or undersampling is not required.

x=df.drop('Class',axis=1)
y=df['Class']

# Splitting the Data

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

# Scaling the data

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

# Applying PCA

pca=PCA(n_components=0.95,random_state=42)
x_train_pca=pca.fit_transform(x_train_scaled)
x_test_pca=pca.transform(x_test_scaled)

print("Original training data shape:", x_train_scaled.shape)
print("PCA transformed training data shape:", x_train_pca.shape)
print("Original test data shape:", x_test_scaled.shape)
print("PCA transformed test data shape:", x_test_pca.shape)

# Visualizing the top 2 principal components
plt.figure(figsize=(8, 6))
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Genomic Data Projection')
plt.colorbar(label='Cancer Stage')
plt.grid(True)
plt.show()

from sklearn.metrics import accuracy_score,classification_report

# **Model Creating**


# LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=1000)
lr.fit(x_train_pca,y_train)

y_pred1=lr.predict(x_test_pca)

accuracy=accuracy_score(y_pred1,y_test)
print(f"Accuracy: {accuracy}")

report=classification_report(y_test,y_pred1)
print("Classification Report:")
print(report)

# Confusion Matrix
class_names=['Class 1','Class 2','Class 3','Class 4','Class 5']

cm=confusion_matrix(y_test, lr.predict(x_test_pca))
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# KNN

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_pca,y_train)

y_pred2=knn.predict(x_test_pca)

accuracy=accuracy_score(y_pred2,y_test)
print(f"Accuracy: {accuracy}")

# Plotting Accuracy vs k to find the best value of k
k_vals = range(1, 21)
train_accuracies = []

for k in k_vals:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train_pca, y_train)
    train_accuracies.append(model.score(x_train_pca, y_train))

plt.plot(k_vals, train_accuracies, label='Train Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs k')
plt.legend()
plt.grid(True)
plt.show()

The above graph shows that k=5 gives the highest accuracy

# Using k=5
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_pca,y_train)

y_pred2=knn.predict(x_test_pca)

accuracy=accuracy_score(y_pred2,y_test)
print(f"Accuracy: {accuracy}")

report=classification_report(y_test,y_pred2)
print("Classification Report:")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for KNN")
plt.show()

# SUPPORT VECTOR MACHINE

from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train_pca,y_train)

# Tuning hyperparameters using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(x_train_pca, y_train)

print("Best params:", grid.best_params_)
print("Best score:", grid.best_score_)

svm=SVC(kernel='rbf',C=10,gamma=0.01)
svm.fit(x_train_pca,y_train)

y_pred3=svm.predict(x_test_pca)

accuracy=accuracy_score(y_pred3,y_test)
print(f"Accuracy: {accuracy}")

report=classification_report(y_test,y_pred3)
print("Classification Report:")
print(report)

# Confusion matrix

cm=confusion_matrix(y_test, svm.predict(x_test_pca))
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# XGBoost

from xgboost import XGBClassifier
xgb=XGBClassifier()

# XGBoost treats class labels as index numbers; so they must start at 0.
y_train_xgb=y_train-1
y_test_xgb=y_test-1
xgb.fit(x_train_pca,y_train_xgb)

y_pred4=xgb.predict(x_test_pca)

accuracy=accuracy_score(y_pred4,y_test_xgb)
print(f"Accuracy: {accuracy}")

report=classification_report(y_test_xgb,y_pred4)
print("Classification Report:")
print(report)

# Confusion matrix

cm=confusion_matrix(y_test_xgb, xgb.predict(x_test_pca))
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train_pca,y_train)

y_pred5=rf.predict(x_test_pca)

accuracy=accuracy_score(y_pred5,y_test)
print(f"Accuracy: {accuracy}")

# Using GridsearchCV to find best max_depth and n_estiamtors
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10, None]
}

grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(x_train, y_train)

print("Best parameters for Random Forest:", grid_rf.best_params_)
print("Best score for Random Forest:", grid_rf.best_score_)

rf_new=RandomForestClassifier(n_estimators=100,max_depth=None,random_state=42,criterion='gini')
rf_new.fit(x_train_pca,y_train)

y_pred5=rf_new.predict(x_test_pca)

accuracy=accuracy_score(y_pred5,y_test)
print(f"Accuracy: {accuracy}")

report=classification_report(y_test,y_pred5)
print("Classification Report:")
print(report)

# Confusion matrix

cm=confusion_matrix(y_test, rf.predict(x_test_pca))
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Using Cross-Validation

from sklearn.model_selection import cross_val_score

models = {
    'Logistic Regression': lr,
    'SVM': svm,
    'KNN': knn,
    'Random Forest': rf,
    'XGBoost': xgb
}

# Create a 0-indexed version of the target variable for XGBoost cross-validation
y_xgb_cv=y-1

for name, model in models.items():
    if name=='XGBoost':
        scores=cross_val_score(model,x,y_xgb_cv,cv=5)  # Use 0-indexed labels for XGBoost
    else:
        scores=cross_val_score(model,x,y,cv=5)  # Use original labels for other models
    print(f'{name}: Mean Accuracy={scores.mean():.4f}, Std Dev = {scores.std():.4f}')

#Using graphs to compare the metric values of each algorithm


# Plotting the accuracies of each model

model_names=["Logistic Regression","SVM","K-nearest neighbors","XGBoost","Random Forest"]
accuracy=[0.92307,0.92307, 0.94017, 0.9145, 0.92307]

plt.figure(figsize=(8,5))
bars=plt.bar(model_names,accuracy,color=['skyblue','pink','orange','lightgreen','salmon'])

# Add accuracy values to the plot
for bar in bars:
    yval=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2.0,yval,str(yval),va='bottom',ha='center')

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8,1.0)

plt.tight_layout()
plt.show()

# Plotting the mean accuracies from cross-validation with standard deviation

models=["Logistic Regression","SVM","KNN","Random Forest","XGBoost"]
mean_accuracies=[0.9333,0.9333,0.9077,0.9282,0.9103]
std_devs=[0.0366,0.0297,0.0096,0.0208,0.0397]

plt.figure(figsize=(10,6))
bars=plt.bar(models,mean_accuracies,yerr=std_devs,capsize=6,color='skyblue',edgecolor='black')

# Annotate bars with values
for bar in bars:
    yval=bar.get_height()
    plt.text(bar.get_x()+bar.get_width()/2,yval + 0.002,f"{yval:.4f}",ha='center',va='bottom')

plt.ylabel('Mean Accuracy (Cross-Validation)')
plt.ylim(0.88,0.98)
plt.title('Model Comparison: Mean Accuracy with Standard Deviation')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
