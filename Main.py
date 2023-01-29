import os
import time
import unicodedata
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from pandas_datareader import data as pdr
import fix_yahoo_finance
from Prepare1 import *

class Main(object):

	def __init__(self):
		self.data = None
		self.X_train = None
		self.y_train = None
		self.X_test = None
		self.y_test = None
		self.model = None
	
	def prepare_X_y(self, data):
		X = data.values
		ind = list(data.columns).index('Open')
		y = []
		for i in range(X.shape[0]-1):
			if (X[i+1,ind]-X[i,ind])>0:
				y.append(1)
			else:
				y.append(0)
		y = np.array(y)
		X = X[:-1]
		return X,y
	
	def split_train_test(self,X,y):
		split_ratio=0.9
		train_size = int(round(split_ratio * X.shape[0]))
		X_train = X[:train_size]
		y_train = y[:train_size]
		X_test = X[train_size:]
		y_test = y[train_size:]

		print(X_train.shape, y_train.shape)
		print(X_test.shape, y_test.shape)
		return X_train, X_test, y_train, y_test
		
	def class_balance(self,train):
		count_class_0, count_class_1 = train['target'].value_counts()
		train_class_0 = train[train['target'] == 0]
		train_class_1 = train[train['target'] == 1]

		if count_class_0>count_class_1:
			train_class_0_under = train_class_0.sample(count_class_1)
			train_sampled = pd.concat([train_class_0_under, train_class_1], axis=0)
		else:
			train_class_1_under = train_class_1.sample(count_class_0)
			train_sampled = pd.concat([train_class_0, train_class_1_under], axis=0)
		
		print(train_sampled['target'].value_counts())
		train_sampled['target'].value_counts().plot(kind='bar', title='Count (target)')
		plt.show()
		return train_sampled
	
	def train_model(self, X_train, y_train):
		model = xgb.XGBClassifier()
		model.fit(X_train, y_train)
		return model
	
	def predict(self, model, X_test, y_test):
		y_pred = model.predict(X_test)
		y_pred = [round(value) for value in y_pred]
		accuracy = accuracy_score(y_test, y_pred)
		f1=f1_score(y_test,y_pred)
		print("F1_score: %.2f%%" % (f1 * 100.0))
		print("Accuracy: %.2f%%" % (accuracy * 100.0))
	
	def plot_feature_imp(self, data, model):
		imp_score = pd.DataFrame(model.feature_importances_, columns=['Importance Score'])
		features = pd.DataFrame(data.columns, columns=['Features'])
		feature_imp = pd.concat([features,imp_score], axis=1)
		print(feature_imp)
		# feature_imp = feature_imp.sort_index(by='Importance Score', ascending=False)
		plt.figure(figsize=(20,7))
		sns.barplot(x=feature_imp['Importance Score'], y=feature_imp['Features'])
		plt.show()
		
if __name__ == '__main__':
	Prepare_data = Prepare_data()
	main = Main()
	main.data = Prepare_data.data
	X,y = main.prepare_X_y(main.data)
	main.X_train, main.X_test, main.y_train, main.y_test = main.split_train_test(X,y)
	main.model = main.train_model(main.X_train, main.y_train)
	main.predict(main.model, main.X_test, main.y_test)
	main.plot_feature_imp(main.data, main.model)