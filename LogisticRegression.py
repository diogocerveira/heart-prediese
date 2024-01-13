import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv('heart.csv')

# Verificar se existem "null values" no dataset
if data.isnull().sum().any():
    print("Missing values in the dataset")

# Separar as Features e o Target
X=data.drop(columns='target', axis=1)
Y=data['target']

# Dividir os dados em training data & Test data, de forma aleatória
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

#Logistic Regression
model = LogisticRegression()

#Treinar o modelo Logistic Regression com a Training Data
model.fit(X_train, Y_train)

# Avaliação do Modelo

#Accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on Training Data: ', training_data_accuracy)

#Accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on Test Data: ', test_data_accuracy)

# Sistema

input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3) # Estes valores são os da primeira linha do heart.csv

input_data_as_numpy_array = np.asarray(input_data) #Converter a input data num array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) #Reshape do array em uma linha em quantas colunas necessária para preservar o número original de elementos

prediction = model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print('The patient does not have a Heart Disease')
else:
    print('The patient has a Heart Disease')
