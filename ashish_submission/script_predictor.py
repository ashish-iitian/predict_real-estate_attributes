import csv
from sklearn.linear_model import LogisticRegression as logistic
import numpy as np
from script_feature_extractor import feature_extraction

class regression:

	def __init__(self,path):
		self.path = path	

	def classifier(self, cat_features, text_features, label):

		''' applies scikit-learn's logistic regression algorithm to learn relation between feature matrix and output label '''

		features = np.append(cat_features,text_features,1)	#augment the features to form a feature matrix
		return logistic(C=1.0).fit(features, label)

	def predictor(self,logit):

		''' parses the test data set and makes predictions based on the learning of logistic regression and returns the predicted label '''

		f = open(self.path,'rb')
		test_file = csv.reader(f, delimiter = '\t')
		test_file = list(test_file)[1:]
		length = len(test_file)
		print length
		extract_test = feature_extraction('test','./data/cat_filter.pk','./data/business_vec.pk','./data/descrp_vec.pk')
		test_lic_type = [x[0] for x in test_file]
		test_sub_type = [x[4] for x in test_file]
		enc_test_cat_feat = extract_test.process_categorical_features(test_lic_type, test_sub_type,'test',length)
		test_business_name = [x[1] for x in test_file]
		test_descrp_feat = [x[2]+x[3] for x in test_file]
		enc_test_descrp_feat = extract_test.process_text_features(test_business_name,test_descrp_feat,'test')
		test_features = np.append(enc_test_cat_feat,enc_test_descrp_feat,1)	#augment the features to form a feature matrix
		y_pred = logit.predict(test_features)
		return y_pred
