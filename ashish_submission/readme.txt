
1) pip install -U scikit-learn

2) To run the codes, please use the command:

	python script_classification.py

3) The code has been divided into 3 modules:

 script_classification.py: it behaves as the main() to handle calls to other modules like script_feature_extractor.py and script_predictor.py

4) I used scikit-learn functionalities to work my solution, i.e. process the data set and design the classifier. 

5) I treated the set of features from input dataset as being in 2 groups:

a) categorical: 
License_type and Sub_type features that assume few distinct values only. I used DictVectorizer to implement One-Hot-encoding that transforms the string features into 0/1 feature vectors. 
b) text features: 
Business_name, Legal_description, Description that are treated as Bad-of-Words model. I used CountVectorizer to transform them into 0/1 feature vectors (augmented the feature vectors of Business_name with Legal_description and Description feature vectors)

6) I used the logistic regression algorithm for classification from scikit-learn.
Logistic regression uses stochastic gradient descent method to update its weights and learn from the dataset so that it can make predictions.

7) I ignored the Job_value feature because pre-processing and analysis of training data suggested it had a lot of values missing.

As an improvement, I could have used Scikit.preprocessing.Imputer to deal with those cases.

