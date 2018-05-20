import  pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle


class Main(object):

	def __init__(self,path="classifier/clf.pkl"):
		self.path= path
	
	@classmethod
	def load_clf(self,path):
		return pickle.load(open(path, 'rb'))

	@classmethod
	def load_test(self,loc):
		
		self.df = pd.read_csv(loc, sep=",",engine="python")
		self.df = self.df.fillna('0')

		#label encoding
		lb_make = LabelEncoder()
		heads = self.df.columns
		for i in range(len(self.df.columns)):
			if self.df[heads[i]].dtypes == 'O':
				self.df[heads[i]] = lb_make.fit_transform(self.df[heads[i]].astype(str))

		sc = StandardScaler()
		self.x= sc.fit_transform(self.df)

		return self.x

if __name__ == "__main__":
	
	path="classifier/clf.pkl"
	obj = Main(path)
	clf = obj.load_clf(path)
	test = obj.load_test('Dataset/test.csv')
	output = clf.predict(test)

	print("The result is : {}".format(output))


