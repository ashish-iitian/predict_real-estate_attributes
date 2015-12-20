from sklearn.preprocessing import OneHotEncoder, Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import cPickle as pickle


class feature_extraction:
	
	def __init__ (self,md_type,cat_pkl_path,biz_pkl_path,text_pkl_path):
		self.mode = md_type
		self.cat_pkl_path = cat_pkl_path
		self.business_pkl_path = biz_pkl_path
		self.descrp_pkl_path = text_pkl_path
		
	
	def processor(self, feature_file):

		''' parses the input file to isolate variable to be predicted for learning algorithm and set of features '''

		list_features = []
		var_type = []
		for line in feature_file:
			var_type.append(line[4])
			list_features.append(line[0:4]+line[5:])
		return var_type, list_features

			
	def process_categorical_features(self, lic, sub, mode, length):

		''' converts features like LICENSE_type and SUB_type (that assume some finite values, hence can be treated as categorized) into vectors during training and uses them in test phase ''' 

		d = [{'lic_type':lic[i],'sub_type':sub[i]} for i in xrange(length)]
		if mode == 'train':
			vec = DictVectorizer()
			train_cat_array = vec.fit_transform(d).toarray()		
			with open(self.cat_pkl_path,'wb') as f:
				pickle.dump(vec,f)
			f.close()
			return train_cat_array
		elif mode == 'test':
			with open(self.cat_pkl_path,'rb') as fh:
				vec_test = pickle.load(fh)
				fh.close()
			test_cat_feat = vec_test.transform(d).toarray()
			return test_cat_feat


	def process_text_features(self, business, text, mode):

		''' converts business_name, legal_descrp, descrp features (that can't be categorized into few values) into vectors during training and uses them in test phase ''' 

		if mode == 'train':
			vec_business = CountVectorizer(min_df=2)
			train_business_array = vec_business.fit_transform(business).toarray()
			vec_descrp = CountVectorizer(min_df=2)			
			train_descrp_array = vec_descrp.fit_transform(text).toarray()
			with open(self.business_pkl_path,'wb') as f:
				pickle.dump(vec_business,f)
			f.close()			
			with open(self.descrp_pkl_path,'wb') as f:
				pickle.dump(vec_descrp,f)
			f.close()
			train_text_array = np.append(train_business_array,train_descrp_array,1)
			return train_text_array
		elif mode == 'test':
			with open(self.business_pkl_path,'rb') as fh:
				vec_business = pickle.load(fh)
				fh.close()
			with open(self.descrp_pkl_path,'rb') as fh:
				vec_descrp = pickle.load(fh)
				fh.close()
			test_business_name = vec_business.transform(business).toarray()
			test_descrp_feat = vec_descrp.transform(text).toarray()
			test_text_feat = np.append(test_business_name,test_descrp_feat,1)
			return test_text_feat			
	
		
