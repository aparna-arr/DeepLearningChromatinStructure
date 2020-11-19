import math
import numpy as np
import sys
import os
import subprocess
import time
from sklearn.metrics import roc_auc_score
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1)
class RandomForest():
	def __init__(self, data, architecture, tag, num_estimators = 1000, min_sample_split = 20, max_depth = None, max_leaf_nodes = 4, random_state = 0, class_weight="balanced", n_jobs = -1, print_cost = True):
		self.tag = tag
		self.X_train = data['X_train']
		self.Y_train = data['Y_train']
		self.X_test = data['X_dev']
		self.Y_test = data['Y_dev']
		self.architecture = architecture
		self.num_estimators = num_estimators
		self.min_sample_split = min_sample_split
		self.max_depth = max_depth
		self.max_leaf_nodes = max_leaf_nodes
		self.random_state = random_state
		self.class_weight = class_weight
		self.n_jobs = n_jobs
		self.print_cost = True
		(self.m,_) = self.X_train.shape
		self.n_y = self.Y_train.shape[0]

		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		self.savestr = self.tag + "_" + self.architecture + "_" + self.timestr
		self.savedir = 'save/' + self.savestr + '/'
		self.logfile = self.savestr + ".log"
		self.predfile = self.savestr + ".predictions"
		self.costpdf = self.savestr + "-Cost.pdf"

	def run(self):
		model = ensemble.RandomForestClassifier(
			n_estimators = self.num_estimators, 
			min_samples_split = self.min_sample_split, 
			max_depth = self.max_depth, 
			random_state = self.random_state, 
			max_leaf_nodes = self.max_leaf_nodes,
			class_weight = self.class_weight,
			n_jobs = self.n_jobs,
			verbose = 1)	
		
		model.fit(self.X_train, self.Y_train)
		train_pred = model.predict_proba(self.X_train)[:,1]
		fpr, tpr, thresholds = roc_curve(self.Y_train, train_pred)
		train_roc_auc = auc(fpr, tpr)

		dev_pred = model.predict_proba(self.X_test)[:,1]

		fpr, tpr, thresholds = roc_curve(self.Y_test, dev_pred)
		dev_roc_auc = auc(fpr, tpr)

		fp = open(self.tag + ".log", "w")
		fp.write("train AUC: " + str(train_roc_auc) + " dev AUC: " + str(dev_roc_auc))

		fp.close()
		return dev_pred, dev_roc_auc, model

