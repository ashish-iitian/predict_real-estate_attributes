import csv
import cPickle as pickle
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.metrics import classification_report
from buildzoom_feature_extractor import feature_extraction
from buildzoom_predictor import regression


def train(length,train_data):

	''' training data set is sent to be processed and to extract label and feature vectors which will be returned to the main'''
	
	extractor = feature_extraction('train','./data/cat_filter.pk','./data/business_vec.pk','./data/descrp_vec.pk')	
	type_val, parsed_input = extractor.processor(train_data)
	lic_type = [x[0] for x in parsed_input]
	sub_type = [x[4] for x in parsed_input]
	enc_cat_features = extractor.process_categorical_features(lic_type, sub_type,'train', length)
	descrp = [x[2]+x[3] for x in parsed_input]
	business = [x[1] for x in parsed_input]
	enc_text_features = extractor.process_text_features(business,descrp,'train')	
	return type_val, enc_cat_features, enc_text_features



def test(path, categorical_feat, text_feat, type_value):

	''' the test data set is used to call logistic regression module that learns the features and makes predictions which are returned to the main function '''

	f = open(path,'rb')
	log_reg = regression(path)
	logit = log_reg.classifier(categorical_feat,text_feat,type_value)
	y_pred = log_reg.predictor(logit)
	return y_pred	



def main():

	''' handles the function calls and saves the prediction of the learning algorithm '''

	f = open('./data/train_data.csv','rb')
	data_file = csv.reader(f, delimiter = '\t')
	train_file = list(data_file)[1:]
	length = len(train_file)
	f.close()
	label, enc_cat_features, enc_text_features = train(length,train_file)
	enc_label = []
	for i in xrange(length):
		if label[i] == 'ELECTRICAL':
			enc_label.append(1)
		else:
			enc_label.append(0)
	Y = np.array(enc_label).reshape(length,1)
	y_pred = test('./data/xtest_data.csv', enc_cat_features, enc_text_features, Y)
	with open('./data/ytest_pred.csv','wb') as fh:
		writer = csv.writer(fh, delimiter='\n')
		writer.writerow(y_pred)
	fh.close()		


if __name__ == "__main__":
	main()
