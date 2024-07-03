#!/usr/bin/env python

import pandas as pd

### LOAD AND POST-TREAT THE NEW DATA

# We load the data
data_path = "./drug200.csv"
data = pd.read_csv(data_path)

# Checks with print
print(data.head())
print(data.info())
print(data.describe())

# Separate features and the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Pre-processing of the data
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='median')),
	('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
	('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
	transformers=[
		('num', numeric_transformer, numeric_features),
		('cat', categorical_transformer, categorical_features)])

### CREATE THE MODEL FOR PREDICTION

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the model
model = Pipeline(steps=[
	('preprocessor', preprocessor),
	('classifier', SVC(probability=True, random_state=42))])

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize parameters with Bayesian Optimization

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# We define the research space for the bayesian optimization
param_grid = {
	'classifier__C': Real(1e-6, 1e+6, prior='log-uniform'),
	'classifier__gamma': Real(1e-6, 1e+1, prior='log-uniform'),
	'classifier__kernel': Categorical(['linear', 'rbf', 'poly'])
}

# Optimization with cross-validation set to 5
opt = BayesSearchCV(model, param_grid, n_iter=50, cv=5, n_jobs=-1, random_state=42)

# Train the new model with the optimal parameters
opt.fit(X_train, y_train)

# Print the best parameters
print(f'Best parameters: {opt.best_params_}')
best_model = opt.best_estimator_

# New evaluation of the model
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

y_pred_optimized = best_model.predict(X_test)
print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_optimized)}')
print('Optimized Classification Report:')
print(classification_report(y_test, y_pred_optimized))
print('Optimized Confusion Matrix:')
cm_optimized = confusion_matrix(y_test, y_pred_optimized)
sns.heatmap(cm_optimized, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
#plt.show()
plt.savefig("Confusion_Matrix.png", dpi=600, bbox_inches="tight")

# Save the best model and the preprocessor
import joblib
joblib.dump(best_model, 'svm_model.joblib')
#joblib.dump(preprocessor, 'preprocessor.joblib')
