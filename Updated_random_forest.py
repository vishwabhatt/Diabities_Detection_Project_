import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load dataset
dataset = pd.read_csv('/content/diabetes_prediction_dataset 2.csv')

print("Missing values per column:")
print(dataset.isnull().sum())

# One-hot encode categorical variables
data = pd.get_dummies(dataset, columns=['gender', 'age','hypertension','heart_disease','smoking_history','bmi','HbA1c_level','blood_glucose_level'])

# Splitting features and target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for faster tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [50, 100, 200],  # Fewer trees for faster execution
    'max_depth': [10, 20, 30],  # Limit depth for faster processing
    'min_samples_split': [2, 5],  # Standard options for split
    'min_samples_leaf': [1, 2],  # Minimum number of samples at a leaf node
    'max_features': ['auto', 'sqrt']  # Features to consider for split
}

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Use RandomizedSearchCV for faster hyperparameter tuning
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Train the model using RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters and best estimator from random search
best_params = random_search.best_params_
best_rf_model = random_search.best_estimator_

print("Best parameters found through RandomizedSearchCV:", best_params)

# Predict on test set using the fine-tuned model
y_pred_rf = best_rf_model.predict(X_test)

# Calculate accuracy of the fine-tuned Random Forest model
accuracy_rf = best_rf_model.score(X_test, y_test)
print(f"Fine-tuned Random Forest Accuracy: {accuracy_rf * 100:.2f}%")

# Ask for user input for gender
user_gender = input("Enter your gender (Male/Female/Both): ").strip().capitalize()

# Ask user if they want to search by specific age or age range
age_option = input("Do you want to search by specific age or age range? (Enter 'specific' or 'range'): ").strip().lower()

if age_option == "specific":
    # Ask for specific age
    user_age = float(input("Enter the specific age you want to filter by: "))

    # Filter the dataset based on gender and specific age
    if user_gender == "Both":
        filtered_data = dataset[(dataset['age'] == user_age)]
    else:
        filtered_data = dataset[(dataset['gender'].str.strip().str.capitalize() == user_gender) &
                                (dataset['age'] == user_age)]
else:
    # Ask for age range
    print("Enter the age range you want to filter (e.g., 20 to 50):")
    age_lower = float(input("Enter the lower age limit: "))
    age_upper = float(input("Enter the upper age limit: "))

    # Filter the dataset based on gender and age range
    if user_gender == "Both":
        filtered_data = dataset[(dataset['age'] >= age_lower) & (dataset['age'] <= age_upper)]
    else:
        filtered_data = dataset[(dataset['gender'].str.strip().str.capitalize() == user_gender) &
                                (dataset['age'] >= age_lower) & (dataset['age'] <= age_upper)]

# Count total records for the selected gender and age or age range
total_count = len(filtered_data)

# Count how many have diabetes and how many don't
diabetes_count = len(filtered_data[filtered_data['diabetes'] == 1])
no_diabetes_count = total_count - diabetes_count

# Output the results
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
