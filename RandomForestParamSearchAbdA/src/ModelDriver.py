import sys
from .ReadData import *
from .Model import *
import sklearn.metrics as metrics
import itertools

class ModelDriver():
	def __init__(self, params):
		self.params = params
		self.data = None
		self.model = None

	def cross_validation(self):
		from sklearn.model_selection import StratifiedKFold

		XvalLogFile = self.params['tag'] + "_KfoldXval.log"
	
		dat = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, False, normalize=False).load_data_rf()
		testdat = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_test'], self.params['Y_test'], False, False, normalize=False).load_data_rf()

		num_train = dat['X_train'].shape[0]
		num_dev = dat['X_dev'].shape[0]

		all_X = np.concatenate((dat['X_train'], dat['X_dev']), axis=0)
		all_Y = np.concatenate((dat['Y_train'], dat['Y_dev']), axis=0)

		print("shape all_Y: " + str(all_Y.shape))

		seed = 2

		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)	
		baseTag = self.params['tag']
	
		foldCount = 0

		aucs = list()
		odds = list()

		for train_idx, dev_idx in kfold.split(all_X, all_Y):
			X_train = all_X[train_idx,:]
			Y_train = all_Y[train_idx]
			
			normMean = np.mean(X_train, axis=0)
			normStd = np.std(X_train,axis=0) + 0.000000001
			
			for i in range(X_train.shape[0]):
				X_train[i,:] = np.divide(X_train[i,:] - normMean, normStd)

			X_dev = testdat['X_dev']
			Y_dev = testdat['Y_dev']
			X_dev = np.divide(X_dev - normMean, normStd)

			self.data = {'X_train' : X_train, 'Y_train' : Y_train, 'X_dev' : X_dev, 'Y_dev' : Y_dev}
			self.params['tag'] = baseTag + "_KfoldXval_" + str(foldCount)

			self.init_model()
			sigmoidZ, auc = self.run_model()

			test_y_pred = np.round(sigmoidZ)
			tp_idxs = np.nonzero(np.logical_and(test_y_pred, testdat['Y_dev']))
			tn_idxs = np.nonzero(np.logical_and((1-test_y_pred), (1-testdat['Y_dev'])))
			fn_idxs = np.nonzero(np.logical_and((1-test_y_pred), testdat['Y_dev']))
			fp_idxs = np.nonzero(np.logical_and(test_y_pred, (1-testdat['Y_dev'])))

			tp_idxs = tp_idxs[0]
			tn_idxs = tn_idxs[0]
			fp_idxs = fp_idxs[0]
			fn_idxs = fn_idxs[0]
			
			num_tp = len(tp_idxs)
			num_tn = len(tn_idxs)
			num_fp = len(fp_idxs)
			num_fn = len(fn_idxs)

			odd = (num_tp * num_tn) / (num_fp * num_fn)

			aucs.append(auc)
			odds.append(odd)

			foldCount += 1

		print("KfoldXvals aucs:")
		print(aucs)

		fp = open(XvalLogFile, "w")
		fp.write(",".join([str(x) for x in aucs]) + "\n")
		fp.write(",".join([str(x) for x in odds]) + "\n")
		
		fp.close()

	def load(self):
		if self.params['architecture'].startswith("rf"):
			self.data = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev']).load_data_rf()
	
	def init_model(self):
		if self.params['architecture'].startswith("rf"):
			self.model = RandomForest(self.data, self.params['architecture'], 
				self.params['tag'], 
				num_estimators = self.params['num_estimators'], 
				min_sample_split = self.params['min_sample_split'], 
				max_depth = self.params['max_depth'], 
				max_leaf_nodes = self.params['max_leaf_nodes'],
				random_state = self.params['random_state'], 
				class_weight = self.params['class_weight'], 
				n_jobs = self.params['n_jobs'], 
				print_cost = self.params['print_cost'])	
	def run_model(self):
		return self.model.run()
