
import pandas as pd
import time
import gensim
import sys
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
import sklearn
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
sys.path.insert(0, '..')
from assignment8.my_evaluation import my_evaluation
from sklearn.pipeline import FeatureUnion


class my_model():
	def fit(self, X, y):
		# do not exceed 29 mins
		self.y_data_class = y
		X = self.clean_training_data(X)

		self.preprocessor = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False,ngram_range=(1,5))
		XX = self.preprocessor.fit_transform(X["description"])

		desc = self.preprocessor.fit_transform(X["description"])
		title_transform = self.preprocessor.fit_transform(X["requirements"])
		loc = self.preprocessor.fit_transform(X["description"])
		#rfc = RandomForestClassifier()
		#self.clf = KNeighborsClassifier(n_neighbors = 5)
		
		#knn_grid = {"n_neighbors": [3,5,7],"weights": ["uniform","distance"],"p": [1,2]}
		
		self.clf = SGDClassifier(class_weight = "balanced",shuffle = True, random_state = 25,max_iter=2000,warm_start=True)

		sg_grid = {"penalty": ["l1","l2","elasticnet"]}
		rf_grid = {"max_depth": [10, 15, 25, 35, 45,55,65],
				   "criterion": ['gini', 'entropy'],
				   "min_samples_split": [2, 3, 4, 5],
				    "n_estimators": [100]
				   }
		#self.clf = RandomizedSearchCV(sgd, sg_grid, random_state=0, n_jobs=-1, verbose = 5, cv = 10)
		#self.clf = RandomizedSearchCV(rfc, rf_grid, random_state=0, n_jobs=-1, verbose = 5, cv = 10)
		#self.clf = GridSearchCV(rfc, rf_grid, verbose = 5)
		self.clf.fit(XX, y)
		return

	def predict(self, X):
		# remember to apply the same preprocessing in fit() on test data before making predictions
		X = self.clean_training_data(X)
		#X = self.clean_predict(X)

		#X = self.clean_training_data(X)
		XX = self.preprocessor.transform(X["description"])
		#print(XX)
		#print(XX)
		# rf_grid = {"max_depth": [10, 15, 25, 35, 45],
		# 		   "criterion": ['gini', 'entropy'],
		# 		   "min_samples_split": [2, 3, 4, 5],
		# 		   "n_estimators": [100, 150, 200]
		# 		   }
		# rf_model = RandomizedSearchCV(self.clf, rf_grid, random_state=0, n_jobs=-1)
		predictions = self.clf.predict(XX)
		return predictions

	def clean_training_data(self, data_frame):
	
		self.stopWords = STOPWORDS
		columns = ['location','description', 'requirements','title']
		
		spec_chars = ["!",'"',"#","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","\\","]","^","_","`","{","|","}","~","–","•","$"]
		
		for column in columns:
			for character in spec_chars:
				data_frame[column] = data_frame[column].str.replace(character," ")
				data_frame[column] = data_frame[column].str.split().str.join(" ")

		data_frame.drop('has_questions', axis=1, inplace=True)

		data_frame['telecommuting'] = data_frame.telecommuting.map({1: 't', 0: 'f'})
		data_frame['has_company_logo'] = data_frame.has_company_logo.map({1: 't', 0: 'f'})

		data_cols = list(data_frame.columns.values)

		for column in data_cols:
			self.rem_stopwords(data_frame, column)

		return data_frame

	def rem_stopwords(self, data_frame, column):
		data_frame[column] = data_frame[column].apply(lambda x: " ".join([i for i in x.lower().split() if i not in self.stopWords]))

	def balance_data(self,X, y):
		X_fraud = X.loc[y == 1]
		y_fraud = y.loc[y == 1]

		X_real = X.loc[y == 0]
		y_real = y.loc[y == 0]

		X_real = resample(X_real, n_samples=1000)
		y_real = resample(y_real, n_samples=1000)

		X_balanced = pd.concat([X_real, X_fraud])
		Y_balanced = pd.concat([y_real, y_fraud])


		return X_balanced, Y_balanced
		# print(X_balanced.shape)
		# print(Y_balanced.shape)

	def clean_data(self,data):
		fake = data[data.fraudulent == 1]
		real = data[data.fraudulent == 0]

		num_real, num_fake = data.fraudulent.value_counts()

		real_undersampled = real.sample(num_fake)
		test_undersampled = pd.concat([real_undersampled, fake], axis=0)

		# Y=test_under['fraudulent']
		data_cleaned = test_undersampled.dropna()
		data_cleaned.reset_index(inplace=True)
		test_undersampled['information'] = test_undersampled['title'] + ' ' + test_undersampled['location'] + ' ' + \
										   test_undersampled['description'] + ' ' + test_undersampled['requirements']
		data = test_undersampled.drop(['title', 'description', 'requirements', 'location'], axis=1)
		data = data.drop('has_questions', axis=1)
		data['has_company_logo'].replace({1: "true", 0: "false"}, inplace=True)
		data['fraudulent'].replace({0: "Real", 1: "Fake"}, inplace=True)

		stopwords = gensim.parsing.preprocessing.STOPWORDS

		spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
					  "*", "+", ",", "-", ".", "/", ":", ";", "<",
					  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
					  "`", "{", "|", "}", "~", "–", "•", "$"]

		for char in spec_chars:
			data['information'] = data['information'].str.replace(char, ' ')
			data['information'] = data['information'].str.split().str.join(" ")

		data['information'] = data['information'] + ' ' + data['has_company_logo']
		data = data.drop('has_company_logo', axis=1)
		data = data.drop('telecommuting', axis=1)

		# remove all words like 'in','a','the' from the 'information' column, as those words do not add any significant value to the data.
		data['information'] = data['information'].apply(
			lambda p: " ".join([w for w in str(p).lower().split() if w not in stopwords]))

		return data

	def clean_predict(self,test_undersampled):

		# test_undersampled = test_undersampled.dropna()
		# test_undersampled.reset_index(inplace=True)
		test_undersampled['information'] = test_undersampled['title'] + ' ' + test_undersampled['location'] + ' ' + \
										   test_undersampled['description'] + ' ' + test_undersampled['requirements']
		data = test_undersampled.drop(['title', 'description', 'requirements', 'location'], axis=1)
		# data = data.drop('has_questions', axis=1)
		data['has_company_logo'].replace({1: "true", 0: "false"}, inplace=True)

		# stopwords = gensim.parsing.preprocessing.STOPWORDS
		#
		# spec_chars = ["!", '"', "#", "%", "&", "'", "(", ")",
		# 			  "*", "+", ",", "-", ".", "/", ":", ";", "<",
		# 			  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
		# 			  "`", "{", "|", "}", "~", "–", "•", "$"]
		#
		# for char in spec_chars:
		# 	data['information'] = data['information'].str.replace(char, ' ')
		# 	data['information'] = data['information'].str.split().str.join(" ")
		#
		# data['information'] = data['information'] + ' ' + data['has_company_logo']
		data = data.drop('has_company_logo', axis=1)
		data = data.drop('telecommuting', axis=1)

		# remove all words like 'in','a','the' from the 'information' column, as those words do not add any significant value to the data.
		# data['information'] = data['information'].apply(
		# 	lambda p: " ".join([w for w in str(p).lower().split() if w not in stopwords]))

		return data