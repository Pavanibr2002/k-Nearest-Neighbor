import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

#Load the wine dataset
data=load_wine()
X,y= data.data, data.target

#Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Scale the data
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Create the KNN Classifier
knn= KNeighborsClassifier()

#Hyperparameter tuning with GridSearchCV
param_grid={'n_neighbors':np.arange(1,21)}
grid_search= GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

#Get the best k values from the grid search
best_k= grid_search.best_params_['n_neighbors']
print('Best K value: ',best_k)

#Train the classifier with the best K value on the scaled training data
knn=KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled,y_train)

#make predictions on the scaled test data
y_pred=knn.predict(X_test_scaled)

#calculate the accuracy of the model
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy :" ,accuracy)

#print the classification report
target_names= data.target_names
print('Classification Report: ')
print(classification_report(y_test,y_pred,target_names=target_names))

#visualisations
#bar graph to show the count of each class in the target variables
plt.figure(figsize=(6,4))
sns.countplot(x=y, palette='coolwarm')
plt.xticks(ticks=np.unique(y),labels=target_names,rotation=45)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

#confusion matrix Heatmap
conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='coolwarm',xticklabels=target_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()