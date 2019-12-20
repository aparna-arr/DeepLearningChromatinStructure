import numpy as np
import sys
from scipy.spatial.distance import pdist,squareform
import scipy.spatial as sp

class ReadData():	
	def __init__(self, X_train_file, Y_train_file, X_dev_file, Y_dev_file, return_avg_std_training=False, return_matrix=False):
		self.X_train_file = X_train_file
		self.Y_train_file = Y_train_file
		self.X_dev_file = X_dev_file
		self.Y_dev_file = Y_dev_file
		self.return_avg = return_avg_std_training
		self.return_matrix = return_matrix


	def get_similarity_X(self, origX, class0Norm, class1Norm):
		m = origX.shape[0]
		new_X = np.zeros((m, 2))

		for ex in range(m):
			results0 = 1 - sp.distance.cdist(origX[ex,:,:], class0Norm, 'cosine')
			sim0 = results0[0,1]
			results1 = 1 - sp.distance.cdist(origX[ex,:,:], class1Norm, 'cosine')
			sim1 = results1[0,1]
			new_X[ex,0] = sim0
			new_X[ex,1] = sim1

		return new_X
						

	def load_data(self):
		trainX_orig  = self.load_xyz(self.X_train_file, prev_mean_exist=False, normalize=False)	
		devX_orig = self.load_xyz(self.X_dev_file, prev_mean_exist=False, normalize=False)

		#print("shape of trainX_orig: " + str(trainX_orig.shape))
		#print("shape of devX_orig: " + str(devX_orig.shape))
	
		trainY_orig = self.load_RNA(self.Y_train_file)	
		devY_orig = self.load_RNA(self.Y_dev_file)

		return {'X_train' : trainX_orig, 'Y_train' : trainY_orig, 'X_dev' : devX_orig, 'Y_dev' : devY_orig}
		
#		if self.return_matrix:	
#			trainX_reshaped = trainX_orig.reshape(trainX_orig.shape[0], trainX_orig.shape[1], trainX_orig.shape[2], 1)
#			devX_reshaped = devX_orig.reshape(devX_orig.shape[0],devX_orig.shape[1],devX_orig.shape[2],1)	
#			Y_train = trainY_orig
#			Y_dev = devY_orig
#
#		else:
#			#trainX_reshaped = trainX_orig.reshape(trainX_orig.shape[0], -1).T
#			#devX_reshaped = devX_orig.reshape(devX_orig.shape[0], -1).T	
#
#			class0Norm = np.zeros(prev_mean.shape)
#			class1Norm = np.zeros(prev_mean.shape)
#			for ex in range(trainY_orig.shape[0]):
#				if trainY_orig[ex] == 0:
#					class0Norm += trainX_orig[ex,:,:]
#				else:	
#					class1Norm += trainX_orig[ex,:,:]
#
#			trainX_reshaped = self.get_similarity_X(trainX_orig, class0Norm, class1Norm)
#			devX_reshaped = self.get_similarity_X(devX_orig, class0Norm, class1Norm)
#
#			trainX_reshaped = trainX_reshaped.reshape(trainX_reshaped.shape[0],-1).T
#			devX_reshaped = devX_reshaped.reshape(devX_reshaped.shape[0],-1).T
#
#			Y_train = trainY_orig.T
#			Y_dev = devY_orig.T
#	
#		print("shape of trainX_reshaped: " + str(trainX_reshaped.shape))
#		print("shape of devX_reshaped: " + str(devX_reshaped.shape))
#		print("shape of Y_train: " + str(Y_train.shape))
#		print("shape of Y_dev: " + str(Y_dev.shape))
#
#		if self.return_avg:
#			return {'X_train' : trainX_reshaped, 'Y_train' : Y_train, 'X_dev' : devX_reshaped, 'Y_dev' : Y_dev}, prev_mean, prev_std
#		else:
#			return {'X_train' : trainX_reshaped, 'Y_train' : Y_train, 'X_dev' : devX_reshaped, 'Y_dev' : Y_dev}
	
	def load_xyz(self, xyzfile,prev_mean=None, prev_std=None, prev_mean_exist=False, normalize=False):
		xyzfp = open(xyzfile, "r")
		header = list()
		datDict = dict()
	
		for line in xyzfp:
			if line.startswith('x'):
				header = line.rstrip().split(',')
				continue		
			else:
				x, y, z, barcode, cell = line.rstrip().split(',')	
				cell = int(cell)
				x = float(x)
				y = float(y)
				z = float(z)
				if cell not in datDict:			
					datDict[cell] = dict()
					datDict[cell]['dat'] = [[x,y,z]]
				else:
					datDict[cell]['dat'].append([x,y,z])
	
		xyzfp.close()	
	
		datAr = list()
	
		for c in datDict:
			currDat = datDict[c]['dat']
			currDat = np.array(currDat)
			datAr.append(currDat)
	
		datAr = np.array(datAr)
		
		distance_mats=np.array(list(map(squareform,list(map(pdist,datAr)))))

		#print("shape distance_mats " + str(distance_mats.shape))	
		if normalize == False:
			return distance_mats

		if prev_mean_exist == False:
			normMean = np.mean(distance_mats,axis=0)
			normStd = np.std(distance_mats,axis=0) + 0.000000001

			# uses too much memory for 100k+!
			#distance_mats = np.divide(distance_mats - np.mean(distance_mats,axis=0), np.std(distance_mats,axis=0) + 0.000000001)
			for i in range(distance_mats.shape[0]):
				distance_mats[i,:,:] = np.divide(distance_mats[i,:,:] - normMean, normStd)	
			return distance_mats, normMean, normStd
		else:
			distance_mats = np.divide(distance_mats - prev_mean, prev_std)
			return distance_mats

	def load_RNA(self, rnafile):
		rnafp = open(rnafile, "r")
	
		rnaheader = list()
		rnaDict = dict()
		for line in rnafp:
			if not line[0].isdigit():
				rnaheader = line.rstrip().split(',')
				continue		
			else:
				rna1, cell = line.rstrip().split(',')
				rna1 = int(rna1)
				cell = int(cell)
	
				rnaDict[cell] = dict()
				rnaDict[cell]['dat'] = [rna1]
	
		rnafp.close()
	
		datAr = list()
	
		for c in rnaDict:
			currRNA = rnaDict[c]['dat']
			currRNA = np.array(currRNA)
			datAr.append(currRNA)
		
		datAr = np.array(datAr)
	
		return datAr
