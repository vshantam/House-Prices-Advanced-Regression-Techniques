#loading the modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

class classifier(object):

	def __init__(self,path):

		# load dataset
		self.path = 'Dataset/train.csv'

	@classmethod
	def load_data(self):

		#loading dataset
		self.df = pd.read_csv(self.path, sep = ",", engine = "python")
		#taking care of null values
		self.df = self.df.fillna('0')
		
		return self.df
	
	@classmethod
	def encoding(self):

		#label encoding
		lb_make = LabelEncoder()
		heads = self.df.columns
		for i in range(len(df.columns)):
			if self.df[heads[i]].dtypes == 'O':
				self.df[heads[i]] = lb_make.fit_transform(self.df[heads[i]].astype(str))
	
		return self.df

	@staticmethod
	def load_x_y():
		
		#extracting input and output features
		X = self.df.iloc[:,:-1].values
		Y = self.df.iloc[:,-1].values
	
		return X,Y

