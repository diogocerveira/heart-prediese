import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data=pd.read_csv('heart.csv')

# Verificar se existem "null values" no dataset
if data.isnull().sum().any():
    print("Missing values in the dataset")

# Separar as Features e o Target
X=data.drop(columns='target', axis=1)
Y=data['target']

# Dividir os dados em training data & Test data, de forma aleatória
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Standardização das features
#KNN é um algoritmo baseado na distância, por isso deve-se fazer Standardização
standard_X=StandardScaler()
X_train=standard_X.fit_transform(X_train)
X_test=standard_X.fit_transform(X_test)

# Verificar qual é o melhor valor de k (nº de neighbors)
error = []
# Calcular o erro para K's entre 1 e 30
for i in range(1, 30):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, Y_train)
    pred_i = knn_model.predict(X_test)
    error.append(np.mean(pred_i != Y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate (K Value)')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for K in range(1,20):

    model = KNeighborsClassifier(n_neighbors=K, metric='euclidean')

    # Train the model using the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = model.predict(X_test)

    print(Y_pred)

    # Calculate evaluation metrics
    # pos_label referes to the HD presence class (the positive class)
    pos_label = 1
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, pos_label=pos_label)
    recall = recall_score(Y_test, Y_pred, pos_label=pos_label)
    f1 = f1_score(Y_test, Y_pred, pos_label=pos_label)

    # Calculate the confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)

    # Append the evaluation metrics to the respective lists
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    # Print the evaluation metrics for the current iteration
    print("K:", K)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, columns=["Predicted HD Absence", "Predicted HD Presence"], index=["Actual HD Absence", "Actual HD Presence"]))
    print()


# Create a DataFrame to store the evaluation metrics for each iteration
metrics_df = pd.DataFrame({
    "K": range(1, 20),
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1 Score": f1_list
})

# Print the metrics table
print("Metrics Table:")
print(metrics_df)