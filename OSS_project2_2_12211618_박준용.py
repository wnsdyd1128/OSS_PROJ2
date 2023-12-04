import pickle

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tools.eval_measures import rmse

def sort_dataset(dataset_df):
	return dataset_df.sort_values(by='year', ascending=True)

def split_dataset(dataset_df):
	(X_train, Y_train) = (dataset_df[:1718].drop('salary', axis=True), dataset_df[:1718]['salary'])
	Y_train = Y_train * 0.001
	(X_test, Y_test) = (dataset_df[1718:].drop('salary', axis=True), dataset_df[1718:]['salary'])
	Y_test = Y_test * 0.001
	return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
	return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
	decision_tree_model_file = "decision_tree_regressor.pkl"
	try:
		with open(decision_tree_model_file, "rb") as f:
			decision_tree_model = pickle.load(f)
	except Exception as e:
		decision_tree_model = DecisionTreeRegressor(min_samples_split=3, min_samples_leaf=6)
		decision_tree_model.fit(X_train, Y_train)
	y_pred = decision_tree_model.predict(X_test)

	with open(decision_tree_model_file, "wb") as f:
		pickle.dump(decision_tree_model, f)
	return y_pred

def train_predict_random_forest(X_train, Y_train, X_test):
	random_forest_model_file = "random_forest_regressor.pkl"
	try:
		with open(random_forest_model_file, "rb") as f:
			random_forest_model = pickle.load(f)
	except Exception as e:
		random_forest_model = RandomForestRegressor()
		random_forest_model.fit(X_train, Y_train)
	y_pred = random_forest_model.predict(X_test)

	with open(random_forest_model_file, "wb") as f:
		pickle.dump(random_forest_model, f)
	return y_pred


def train_predict_svm(X_train, Y_train, X_test):
	svm_regessor_model_file = "random_forest_regressor.pkl"
	try:
		with open(svm_regessor_model_file, "rb") as f:
			svm_model = pickle.load(f)
	except Exception as e:
		svm_model = SVR()
		svm_model.fit(X_train, Y_train)
	y_pred = svm_model.predict(X_test)
	with open(svm_regessor_model_file, "wb") as f:
		pickle.dump(svm_model, f)
	return y_pred

def calculate_RMSE(labels, predictions):
	return rmse(labels, predictions)

if __name__=='__main__':
	#DO NOT MODIFY THIS FUNCTION UNLESS PATH TO THE CSV MUST BE CHANGED.
	data_df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')
	
	sorted_df = sort_dataset(data_df)	
	X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)
	
	X_train = extract_numerical_cols(X_train)
	X_test = extract_numerical_cols(X_test)

	dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
	rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
	svm_predictions = train_predict_svm(X_train, Y_train, X_test)
	
	print ("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))	
	print ("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))	
	print ("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))