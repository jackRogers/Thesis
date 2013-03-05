#Jack Rogers
#Thesis 2012
#New College of Florida
#jack.rogers@ncf.edu
#954-376-2345

##########################
### PROGRAM DESCRPTION ###
##########################
"""
This program tests 8 machine learning classifiers on two neurological
datasets using three differnt dimensionality reduction techniques and 
random sampling to generate mean classifcation accuracies
"""

#############
### TO DO ###
#############

#allow comparison between ica and pca with differnt amounts of components
#confusion matrix
#lda scatter plots
#abstract interface for querying results
#unit tests
#finish filling in documentation
#comment complex code more heavily


######################
### IMPORT MODULES ###
######################
import math
import numpy
import scipy.io  
import sklearn											#to load in the mat file
import unittest

from sklearn import decomposition  						#import decomposition for PCA
from sklearn.lda import LDA		    					#import linear discriminant analysis
from sklearn.qda import QDA								#import quadractic discriminatnt analysis

from sklearn import svm									#to use the svm class
from sklearn.neighbors import KNeighborsClassifier  	#import KNN
from sklearn.cluster import KMeans						#import kmeans clustering
from sklearn.naive_bayes import GaussianNB				#import gaussian naive bayes
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression		#import logistic regression
from sklearn.ensemble import RandomForestClassifier 	#import random forests

from sklearn.cross_validation import cross_val_score	#import cross validation

	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""

	

###################################
### DATA MANIPULATION FUNCTIONS ###
###################################

def array_to_list(A):
	"""
	DESCRIPTION:
	This function takes a numpy array and converts it to a standard
	python list
	
	INPUTS:
	A numpy array
	
	OUTPUTS:
	A python list
	
	EXAMPLE USAGE:
	list_object = array_to_list(array_object)
	
	"""
	
	x = []
	for i in range(len(A)):
		x.append([])
		for j in A[i]:
			x[i].append(j)
	return x

def horzcat(A,B):
	"""
	DESCRIPTION:
	This function horizontally concatenates two matrices, making a
	(N x A) matrix and a (N x B) matrix become a (N x A+B) matrix.
	
	INPUTS:
	Two numpy arrays
	
	OUTPUTS:
	One numpy array
	
	EXAMPLE USAGE:
	AnB = horzcat(A,B)
	
	"""
	
	AList = array_to_list(A)
	BList = array_to_list(B)
	for i in range(len(AList)):
		for j in range(len(BList[0])):
			AList[i].append(BList[i][j])
	return numpy.array(AList)

def vertcat(A,B):
	"""
	DESCRIPTION:
	This function veritcally concatenates two matrices, making a 
	(N x C) matrix and a (M x C) matrix of size of size (N+M x C)
		
	INPUTS:
	Two numpy arrays
	
	OUTPUTS:
	One numpy array
	
	EXAMPLE USAGE:
	AnB = vertcatcat(A,B)
	
	"""
	
	AList = array_to_list(A)
	BList = array_to_list(B)
	for i in range(len(BList)):
		AList.append(BList[i])
	return numpy.array(AList)
	
def horzcat_target(A,B):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	for i in range(len(B)):
		A.append(B[i])
	return A

def shuffle_inplace(a):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	p = numpy.random.permutation(len(a))
	return a[p]

def shuffle_in_unison_inplace(a, b):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	assert len(a) == len(b)
	p = numpy.random.permutation(len(a))
	return a[p], b[p]

def make_target(size,val):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	x = []
	for i in range(size):
		y = val
		x.append(y)
	return x

def random_sample(A,percent_training_size):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	shuffled = shuffle_inplace(A)
	start = len(shuffled)*0.01*percent_training_size
	start = int(start)
	Train = shuffled[:start]
	Test = shuffled[start:]
	TrainSize = len(Train)
	TestSize = len(Test)
	return Train,Test,TrainSize,TestSize

##########################
### DEFINE CLASSIFIERS ###
##########################

#classifiers
def run_SVM_linear(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	svc = svm.SVC(kernel='linear')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

def run_SVM_RBF(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	svc = svm.SVC(kernel='rbf')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

def run_KNN(KNN_Neighbors,training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def run_gaussian_naive_bayes(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = GaussianNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def run_bernoulli_naive_bayes(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = BernoulliNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)
	
def run_logistic_regression(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = LogisticRegression()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def run_random_forest(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = RandomForestClassifier(n_estimators=10)
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

#dimensionality reduction
def reduce_LDA(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	lda = LDA(n_components = components)
	return  lda.fit(data,target).transform(data)
	
def reduce_pca(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	pca = decomposition.PCA(n_components=components)
	return pca.fit(data).transform(data)
	
def reduce_ica(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	ica = decomposition.FastICA(n_components=components)
	return ica.fit(data).transform(data)



#################
### LOAD DATA ###
#################

class Matlab_file():
	def __init__(self,Dataset_name,modalities,matlab_dict_keys,class_indices):
		"""
		DESCRIPTION:
		
		
		INPUTS:
		
		
		OUTPUTS:
		
		
		EXAMPLE USAGE:
		
		"""
	
		self.name = name
		self.modalities = modalities
		self.class_indices = class_indices

class Dataset():
	def __init__(self,name,list_of_Matlab_file_objects):
		"""
		DESCRIPTION:
		
		
		INPUTS:
		
		
		OUTPUTS:
		
		
		EXAMPLE USAGE:
		
		"""
	
		self.name = name
		self.matlab_files = list_of_matlab_file_objects
		self.modalities = 
	

	
##################
### UNIT TESTS ###
##################


class TestFunctions(unittest.TestCase):

	#def cellular_example_test(self):
		#"""Tests one_tick() removing the tasks and destinations of cells done stopping."""
		#c = Cell(0,0)
		#c.task = 'stop'
		#c.one_tick()
		#self.assertEquals(c.task,None)
		#self.assertEquals(c.destination,None)
		
	#array_to_list
	def test_array_to_list(self):
		"""Tests the array to list function"""
		pass
			
	#horzcat
	
	#vertcat
	
	#horzcat_target
	
	#shuffle_inplace
	
	#shuffle_in_unison
	
	#make_target
	
	#random_sample
	
	#run_SVM_linear
	
	#run_SVM_RBF
	
	#run_KNN
	
	#run_gaussian_naive_bayes
	
	#run_bernoulli_naive_bayes
	
	#run_logistic_regression
	
	#run_random_forest
	
	#reduce_LDA
	
	#reduce_pca
	
	#reduce_ica
	
	#load_dataset
	
	
