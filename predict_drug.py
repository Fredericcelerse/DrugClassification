#!/usr/bin/env python

import numpy as np
import pandas as pd

# Load the trained model and the preprocessor
import joblib
model = joblib.load('svm_model.joblib')
#preprocessor = joblib.load('preprocessor.joblib')

# Input from the user
def get_user_input():
	print("Enter the desired features:")
	age = float(input("Age: "))
	sex = input("Sex (M/F); ")
	bp = input("Blood Pressure (HIGH/LOW/NORMAL): ")
	cholesterol = input("Cholesterol (HIGH/LOW): ")
	na_to_k = float(input("Na_to_K: "))

	user_data = pd.DataFrame({
		'Age': [age],
		'Sex': [sex],
		'BP': [bp],
		'Cholesterol': [cholesterol],
		'Na_to_K': [na_to_k]
	})

	return user_data

# Request the data from the user
user_data = get_user_input()

# Pre-process the data
#user_data_preprocessed = preprocessor.transform(user_data)

# Predictions
#y_pred_prob = model.predict_proba(user_data_preprocessed)
y_pred_prob = model.predict_proba(user_data)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print the predictions and the uncertainty
medicines = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']
predicted_medicine = medicines[y_pred[0]]
print(f"Prediction: {predicted_medicine}")
print(f"Probability: {y_pred_prob[0]}")


