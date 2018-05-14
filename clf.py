#loading the modules
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


class classifier(object):

	def __init__(self,path):

		# load dataset
		self.path = path


	@classmethod
	def load_data(self,path):

		#loading dataset
		self.df = pd.read_csv(path, sep = ",", engine = "python")
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

	@classmethod
	def load_x_y(self):
		
		#extracting input and output features
		self.X = self.df.iloc[:,:-1].values
		self.Y = self.df.iloc[:,-1].values
	
		return self.X,self.Y

	@classmethod
	def scale(self):

		sc = StandardScaler()
		self.X = sc.fit_transform(self.X)
		
		return self.X

	@classmethod
	def model(self):
	
		self.clf = XGBRegressor()

		self.clf.fit(self.X,self.Y)

		return self.clf

	@classmethod
	def save_model(self):

		return pickle.dump(self.clf, open( "classifier/clf.pkl", "wb" ))

	

if __name__ == "__main__":

	path = 'Dataset/train.csv'

	obj = classifier(path)
	df = obj.load_data(path)
	df= obj.encoding()

	X,Y = obj.load_x_y()
	
	obj.scale()
	obj.model()
	obj.save_model()



