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
def svm_linear(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	svc = svm.SVC(kernel='linear')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

def svm_rbf(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	svc = svm.SVC(kernel='rbf')
	svc.fit(training_data,training_target)
	return svc.score(testing_data,testing_target)

def knn(KNN_Neighbors,training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = KNeighborsClassifier(n_neighbors=5)
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def gnb(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = GaussianNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def bnb(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = BernoulliNB()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)
	
def lr(training_data,training_target,testing_data,testing_target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	clf = LogisticRegression()
	clf.fit(training_data,training_target)
	return clf.score(testing_data,testing_target)

def rf(training_data,training_target,testing_data,testing_target):
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
def lda(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	Lda = LDA(n_components = components)
	return  Lda.fit(data,target).transform(data)
	
def pca(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	Pca = decomposition.PCA(n_components=components)
	return Pca.fit(data).transform(data)
	
def ica(components,data,target):
	"""
	DESCRIPTION:
	
	
	INPUTS:
	
	
	OUTPUTS:
	
	
	EXAMPLE USAGE:
	
	"""
	
	Ica = decomposition.FastICA(n_components=components)
	return Ica.fit(data).transform(data)



###############
### Classes ###
###############

class Matlab_file():
	def __init__(self,mat_path,Dataset_name,modalities_and_keys,classes_and_indices):
		"""
		DESCRIPTION:
		
		
		INPUTS:
		
		
		OUTPUTS:
		
		
		EXAMPLE USAGE:
		
		"""
	
		self.mat_path = mat_path
		self.Dataset_name = Dataset_name
		self.modalities_and_keys = modalities_and_keys
		self.class_indices = classes_and_indices
	

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
		self.class_indices = None
		self.starting_modalities = []
		self.starting_arrays = []
		self.modality_dict = {}
		self.master_target = []
		
		#check matlab file cohesion
		for mat in self.matlab_files:
			#check that all matlab files have the same number of subjects and same class ordering
			if self.class_indices == None:
				self.class_indices = mat.class_indices
			else:
				if self.class_indices == mat.class_indices:
					pass
				else:
					raise ValueError("Matlab files have differnt class indices")
			
			#check that modality is new and not overlapping
			#check that all matlab files have the same dataset name
			#load in new file
			self.load_arrays(mat)
			self.make_combos()
			self.make_master_target()
			
		def load_initial_modalities(self,matlab_object):
			mat_dict = scipy.io.loadmat(matlab_object.mat_path)
			for mod in matlab_object.modalities_and_keys:
				self.starting_modalities.append(mod[0])
				self.starting_arrays.append(mat_dict[mod[1]])
				
		def make_combos(self):
			for i in self.starting_modalities: 
				if len(self.modality_dict.keys()) == 0:
					self.modality_dict[i] = self.starting_arrays[i]
				else:
					for j in self.modality_dict.keys():
						self.modality_dict[i+'+'+j] = horzcat(self.starting_arrays[i],modality_dict[j])
					self.modality_dict[i] = self.starting_arrays[i]
		
		def make_master_target(self):
			for i in range(len(self.class_indices)):
				title = self.class_indices[i][0]
				for j in range(self.class_indices[i][1]):
					self.master_target.append(title)

class Parameters():
	def __init__(self):
		#user defined
		self.N = 1
		self.preprocessing_components = []
		self.percent_training_sizes = []
		self.classifier_parameters = []
		
		#hardcoded
		self.preprocessing_algorithms = []
		self.classifiers = []
	
class Experiment():
	def __init__(self,dataset_object,params):
		self.params = params
		self.dataset_object = dataset_object
		self.result_dict = {}
		#for N iterations
		for n in range(N):
			
			for pre_algo in preprocessing_algorithms:
				
				
				#for a range of preprocessing components
				for component in params.preprocessing_components:
					
					#for each modality
					for modality in dataset_object.modality_dict.keys():
						#preprocess with component set properly
						preprocessed_unsampled_data = eval(pre_algo(component,dataset_object.modality_dict[modality],dataset_object.master_target))
						
						#for each percent_training_size
						for p in params.percent_training_sizes:
							self.create_new_sample(preprocessed_unsampled_data,
							
							#for each of the algorithms
							for classifier in params.classifiers:
								score = eval(classifier(training_data,training_target,testing_data,testing_target))
								#if particular algo is used
								#for classifier_param in self.classifier_parameters:
								
								
	def break_into_class_groups(self,unsampled_data):
		groups = {}
		for i in range(len(self.dataset_object.class_indices)):
			groups[self.dataset_object.class_indices[i][0]] = []
			
			if i == 1:
				for subject in range(self.dataset_object.class_indices[i]):
					groups[self.dataset_object.class_indices[i]].append(unsampled_data[subject])
			else:
				for subject in range(self.dataset_object.class_indices[i-1],self.dataset_object.class_indices[i]):
					groups[self.dataset_object.class_indices[i]].append(unsampled_data[subject])
			return groups
									
	def create_new_sample(self,unsampled_data,modality,p):
		
		groups = self.break_into_class_groups(unsampled_data)
		
			
			
			
		
				
		
		


##############################
### MAIN TESTING FUNCTIONS ###
##############################

def test_main():
	#get inputs
	
	#Dataset A make matlab objects
	Dataset_A_mat1_path = 'E:/Documents/Thesis/Data/fmri_All.mat'
	Dataset_A_mat1_modalities_and_dict_keys = [('FMRI','c')]
	Dataset_A_mat1_class_order_and_amounts = [('Healthy Control',62),('Schizophrenia Subject',54),('Bipolar Disorder Subject',48)]
	Dataset_A_Data_file_1 = Matlab_file(Dataset_A_mat1_path,Dataset_A_mat1_modalities_and_dict_keys,Dataset_A_mat1_class_order_and_amounts)
	
	Dataset_A_mat2_path = 'E:/Documents/Thesis/Data/FA.mat'
	Dataset_A_mat2_modalities_and_dict_keys = [('FA','fa')]
	Dataset_A_mat2_class_order_and_amounts = [('Healthy Control',62),('Schizophrenia Subject',54),('Bipolar Disorder Subject',48)]
	Dataset_A_Data_file_2 = Matlab_file(Dataset_A_mat2_path,Dataset_A_mat2_modalities_and_dict_keys,Dataset_A_mat2_class_order_and_amounts)
	
	#Dataset A	make dataset objects
	Dataset_A_Dataset_name = 'Dataset A'
	Dataset_A_Dataset_matlab_files = [Dataset_A_Data_file_1,Dataset_A_Data_file_2]
	Dataset_A = Dataset(Dataset_A_Dataset_name,Dataset_A_Dataset_matlab_files)
	
	#Dataset B make matlab objects
	
	#Dataset B	make dataset objects
	
	#make experiment object
	Experiment_A = Experiment(Dataset_A)
	
	#make results object
	pass
	
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
	
