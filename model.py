import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('/content/diabetes_prediction_dataset-original.csv')

dataset.head()

print("Missing values per column:")
print(dataset.isnull().sum())

data = pd.get_dummies(dataset, columns=['gender', 'age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])

X = data.drop('diabetes', axis=1) 
y = data['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # With 30% test size and 70% training size
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # With 20% test size and 80% training size

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)

print("Accuracy: %.2f%%" % (accuracy * 100))
