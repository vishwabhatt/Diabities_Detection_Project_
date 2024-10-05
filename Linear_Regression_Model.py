import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression


dataset = pd.read_csv('/content/diabetes_prediction_dataset 4.csv')

print("Missing values per column:")
print(dataset.isnull().sum())


data = pd.get_dummies(dataset, columns=['gender', 'age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])


X = data.drop('diabetes', axis=1)
y = data['diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100))


user_gender = input("Enter your gender (Male/Female/Both): ").strip().capitalize()


age_option = input("Do you want to search by specific age or age range? (Enter 'specific' or 'range'): ").strip().lower()

if age_option == "specific":
   
    user_age = float(input("Enter the specific age you want to filter by: "))

   
    if user_gender == "Both":
        filtered_data = dataset[(dataset['age'] == user_age)]
    else:
        filtered_data = dataset[(dataset['gender'].str.strip().str.capitalize() == user_gender) &
                                (dataset['age'] == user_age)]
else:
   
    print("Enter the age range you want to filter (e.g., 20 to 50):")
    age_lower = float(input("Enter the lower age limit: "))
    age_upper = float(input("Enter the upper age limit: "))

   
    if user_gender == "Both":
        filtered_data = dataset[(dataset['age'] >= age_lower) & (dataset['age'] <= age_upper)]
    else:
        filtered_data = dataset[(dataset['gender'].str.strip().str.capitalize() == user_gender) &
                                (dataset['age'] >= age_lower) & (dataset['age'] <= age_upper)]


total_count = len(filtered_data)


diabetes_count = len(filtered_data[filtered_data['diabetes'] == 1])
no_diabetes_count = total_count - diabetes_count


if total_count > 0:
    if age_option == "specific":
        if user_gender == "Both":
            print(f"Total number of individuals aged {user_age}: {total_count}")
        else:
            print(f"Total number of {user_gender}s aged {user_age}: {total_count}")
    else:
        if user_gender == "Both":
            print(f"Total number of individuals aged between {age_lower} and {age_upper}: {total_count}")
        else:
            print(f"Total number of {user_gender}s aged between {age_lower} and {age_upper}: {total_count}")

    print(f"Number of individuals with diabetes: {diabetes_count}")
    print(f"Number of individuals without diabetes: {no_diabetes_count}")
else:
    if age_option == "specific":
        print(f"No records found for {user_gender}s aged {user_age}.")
    else:
        print(f"No records found for {user_gender}s aged between {age_lower} and {age_upper}.")
