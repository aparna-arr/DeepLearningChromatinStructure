import sys
from .ReadData import *
from .Model import *
import sklearn.metrics as metrics
from .Interpret import *
import itertools

class ModelDriverInterpret():
	def __init__(self, params):
		self.params = params
		self.data = None
		self.train_mean = None
		self.train_std = None
		self.interpretObj = None

	def load(self):
		self.data, self.train_mean, self.train_std = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_test'], self.params['Y_test'], True, True).load_data()	
	
	def init_interpret(self):
		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['minibatch_size'], self.params['threshold_sigmoid'])

	def run(self):
		self.interpretObj.run()
	def run_test(self):
		return self.interpretObj.run_test()

	def make_other_gene_plots(self):
		# " train " = AbdA predictions test
		# "dev/test" = Ubx or AbdB predictions test
		self.interpretObj = Interpret(self.data['X_train'], self.data['Y_train'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['minibatch_size'], self.params['threshold_sigmoid'])				
		
		train_y_pred, train_sigZ = self.run_test()

		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['minibatch_size'], self.params['threshold_sigmoid'])				
		test_y_pred, test_sigZ = self.run_test()

		fprTrain, tprTrain, thresholdTrain = metrics.roc_curve(self.data['Y_train'], train_sigZ)
		fprTest, tprTest, thresholdTest = metrics.roc_curve(self.data['Y_dev'], test_sigZ)

		aucTrain = metrics.roc_auc_score(self.data['Y_train'], train_sigZ)
		aucTest = metrics.roc_auc_score(self.data['Y_dev'], test_sigZ)

		fig = plt.figure()
		plt.title("ROC Curve")
		plt.plot(fprTrain, tprTrain, 'navy', lw = 2, label = 'abdA Train AUC = %0.2f' % aucTrain)
		plt.plot(fprTest, tprTest, 'red', lw=2, label = 'AbdB AUC = %0.2f' % aucTest)
		plt.plot([0, 1], [0, 1], color='r', lw=1, linestyle='--')
		plt.legend(loc = 'lower right')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylabel('TPR')
		plt.xlabel('FPR')
		
		fig.savefig("AbdB_ROC_curve.pdf", bbox_inches='tight')

	def make_plots(self):
		# get training set predictions
		self.interpretObj = Interpret(self.data['X_train'], self.data['Y_train'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['minibatch_size'], self.params['threshold_sigmoid'])				
		
		train_y_pred, train_sigZ = self.run_test()

		# get test set predictions
		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['minibatch_size'], self.params['threshold_sigmoid'])				
		test_y_pred, test_sigZ = self.run_test()

		fprTrain, tprTrain, thresholdTrain = metrics.roc_curve(self.data['Y_train'], train_sigZ)
		fprTest, tprTest, thresholdTest = metrics.roc_curve(self.data['Y_dev'], test_sigZ)

		aucTrain = metrics.roc_auc_score(self.data['Y_train'], train_sigZ)
		aucTest = metrics.roc_auc_score(self.data['Y_dev'], test_sigZ)

		fig = plt.figure()
		plt.title("ROC Curve")
		plt.plot(fprTrain, tprTrain, 'navy', lw = 5, label = 'Train AUC = %0.2f' % aucTrain)
		plt.plot(fprTest, tprTest, 'darkorange', lw = 5, label = 'Test AUC = %0.2f' % aucTest)
		plt.plot([0, 1], [0, 1], color='r', lw=1, linestyle='--')
		plt.legend(loc = 'lower right')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.ylabel('TPR')
		plt.xlabel('FPR')
		
		fig.savefig("ROC_curve.pdf", bbox_inches='tight')

	
		# plot confusion matrix
		cfTrain = metrics.confusion_matrix(self.data['Y_train'], train_y_pred)	

		fig = plt.figure()
		plt.imshow(cfTrain,cmap=plt.cm.Blues,interpolation='nearest')
		plt.colorbar()
		plt.title('Training Confusion Matrix')
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		tick_marks = np.arange(2)
		class_labels = ['0','1']
		plt.xticks(tick_marks,class_labels)
		plt.yticks(tick_marks,class_labels)
		thresh = cfTrain.max() / 2.

		for i,j in itertools.product(range(cfTrain.shape[0]),range(cfTrain.shape[1])):
			plt.text(j,i,format(cfTrain[i,j],'d'),horizontalalignment='center',color='white' if cfTrain[i,j] > thresh else 'black')

		fig.savefig("ConfusionMat_Train.pdf", bbox_inches='tight')
			
		cfTest = metrics.confusion_matrix(self.data['Y_dev'], test_y_pred)	

		fig = plt.figure()
		plt.imshow(cfTest,cmap=plt.cm.Blues,interpolation='nearest')
		plt.colorbar()
		plt.title('Testing Confusion Matrix')
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		tick_marks = np.arange(2)
		class_labels = ['0','1']
		plt.xticks(tick_marks,class_labels)
		plt.yticks(tick_marks,class_labels)
		thresh = cfTest.max() / 2.

		for i,j in itertools.product(range(cfTest.shape[0]),range(cfTest.shape[1])):
			plt.text(j,i,format(cfTest[i,j],'d'),horizontalalignment='center',color='white' if cfTest[i,j] > thresh else 'black')

		fig.savefig("ConfusionMat_Test.pdf", bbox_inches='tight')

		# print average of correctly and incorrectly classified test examples
		tp_idxs = np.nonzero(np.logical_and(test_y_pred, self.data['Y_dev']))
		tn_idxs = np.nonzero(np.logical_and((1-test_y_pred), (1-self.data['Y_dev'])))
		fn_idxs = np.nonzero(np.logical_and((1-test_y_pred), self.data['Y_dev']))
		fp_idxs = np.nonzero(np.logical_and(test_y_pred, (1-self.data['Y_dev'])))
		
		tp_idxs = tp_idxs[0]
		tn_idxs = tn_idxs[0]
		fp_idxs = fp_idxs[0]
		fn_idxs = fn_idxs[0]

		newNormTest = self.data['X_dev']
		normMean = np.mean(newNormTest,axis=0)
		normStd = np.std(newNormTest,axis=0) + 0.000000001
		
		for i in range(newNormTest.shape[0]):
			newNormTest[i,:,:] = np.divide(newNormTest[i,:,:] - normMean, normStd)

		#avg_tp = np.sum(newNormTest[tp_idxs,:,:],axis=0) / len(tp_idxs)
		#avg_tn = np.sum(newNormTest[tn_idxs,:,:],axis=0) / len(tn_idxs)
		#avg_fn = np.sum(newNormTest[fn_idxs,:,:],axis=0) / len(fn_idxs)
		#avg_fp = np.sum(newNormTest[fp_idxs,:,:],axis=0) / len(fp_idxs)

		avg_tp = np.sum(self.data['X_dev'][tp_idxs,:,:],axis=0) / len(tp_idxs)
		avg_tn = np.sum(self.data['X_dev'][tn_idxs,:,:],axis=0) / len(tn_idxs)
		avg_fn = np.sum(self.data['X_dev'][fn_idxs,:,:],axis=0) / len(fn_idxs)
		avg_fp = np.sum(self.data['X_dev'][fp_idxs,:,:],axis=0) / len(fp_idxs)

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_tp[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_TP_Y_test.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_tn[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_TN_Y_test.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_fp[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_FP_Y_test.pdf", bbox_inches = 'tight')
	
		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_fn[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_FN_Y_test.pdf", bbox_inches = 'tight')

		# plot average pos and neg examples train set
		pos_idxs = np.nonzero(self.data['Y_train'])
		neg_idxs = np.nonzero((1-self.data['Y_train']))
		
		pos_idxs = pos_idxs[0]
		neg_idxs = neg_idxs[0]
		
		avg_pos = np.sum(self.data['X_train'][pos_idxs,:,:],axis=0) / len(pos_idxs)
		avg_neg = np.sum(self.data['X_train'][neg_idxs,:,:],axis=0) / len(neg_idxs)

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_pos[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_Pos_Y_train.pdf", bbox_inches = 'tight')
	
		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_neg[:,:,0], cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig("Avg_Neg_Y_train.pdf", bbox_inches = 'tight')

		stdev_pos = np.std(self.data['X_train'][pos_idxs,:,:],axis=0)
		stdev_neg = np.std(self.data['X_train'][neg_idxs,:,:],axis=0)
		stdev_pos = np.ma.masked_where(stdev_pos < 0.0001, stdev_pos)
		stdev_neg = np.ma.masked_where(stdev_neg < 0.0001, stdev_neg)

		cmap = plt.get_cmap("PuOr")
		cmap.set_bad(color='white')
		
		fig = plt.figure()
		plt.imshow(stdev_pos[:,:,0], cmap = cmap, vmin=0.5, vmax=1.5)
		plt.colorbar()
		fig.savefig("Stdev_Pos_Y_train.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		plt.imshow(stdev_neg[:,:,0], cmap = cmap, vmin = 0.5, vmax = 1.5)
		plt.colorbar()
		fig.savefig("Stdev_Neg_Y_train.pdf", bbox_inches = 'tight')
		
		# STDEV test

		#stdev_pos = np.std(newNormTest[tp_idxs,:,:],axis=0)		
		#stdev_neg = np.std(newNormTest[tn_idxs,:,:],axis=0)		
		stdev_pos = np.std(self.data['X_dev'][tp_idxs,:,:],axis=0)		
		stdev_neg = np.std(self.data['X_dev'][tn_idxs,:,:],axis=0)		

		stdev_pos = np.ma.masked_where(stdev_pos < 0.0001, stdev_pos)
		stdev_neg = np.ma.masked_where(stdev_neg < 0.0001, stdev_neg)

		cmap = plt.get_cmap("PuOr")
		cmap.set_bad(color='white')

		fig = plt.figure()
		#plt.imshow(stdev_pos[:,:,0], cmap = 'seismic', norm=norm, vmin=0, vmax=1)
		#plt.imshow(stdev_pos[:,:,0], cmap = 'seismic', vmin=0, vmax=1)
		plt.imshow(stdev_pos[:,:,0], cmap = cmap, vmin=0.5,vmax=1.5)
		plt.colorbar()
		fig.savefig("Stdev_Pos_Y_test.pdf", bbox_inches = 'tight')
	
		fig = plt.figure()
		#plt.imshow(stdev_neg[:,:,0], cmap = 'seismic', norm=norm, vmin=0,vmax=1)
		#plt.imshow(stdev_neg[:,:,0], cmap = 'seismic', vmin=0,vmax=1)
		plt.imshow(stdev_neg[:,:,0], cmap = cmap, vmin=0.5,vmax=1.5)
		plt.colorbar()
		fig.savefig("Stdev_Neg_Y_test.pdf", bbox_inches = 'tight')

class ModelDriver():
	def __init__(self, params):
		self.params = params
		self.data = None
		self.model = None
		self.interpretObj = None

	def init_interpret(self, model_loc, train_mean, train_std, tag, minibatch = 32, thresh = 0.5):
		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], train_mean, train_std, model_loc, tag, minibatch, thresh)

	def run_interpret(self):
		self.interpretObj.run()
	def data_sufficiency(self):
		# assume convolutional network
		percents = [0.05,0.01]

		sufficiencyLogFile = self.params['tag'] + "_dataSufficiency.log"
	
		dat = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, True, normalize=False).load_data()
		
		num_train = dat['X_train'].shape[0]

		idxs = np.arange(num_train)
		np.random.shuffle(idxs)
		baseTag = self.params['tag']
	
		count = 0

		X_dev = dat['X_dev']
		Y_dev = dat['Y_dev']
	
		aucs = list()

		for perc in percents:
			new_train_num = int(num_train * perc)
			curr_idxs = idxs[:new_train_num]

			curr_X_train = dat['X_train'][curr_idxs,:,:,:]
			curr_Y_train = dat['Y_train'][curr_idxs,:]

			normMean = np.mean(curr_X_train, axis=0)
			normStd = np.std(curr_X_train,axis=0) + 0.000000001
			
			for i in range(curr_X_train.shape[0]):
				curr_X_train[i,:,:] = np.divide(curr_X_train[i,:,:] - normMean, normStd)

			curr_X_dev = np.divide(X_dev - normMean, normStd)

			self.data = {'X_train' : curr_X_train, 'Y_train' : curr_Y_train, 'X_dev' : curr_X_dev, 'Y_dev' : Y_dev}
			self.params['tag'] = baseTag + "_dataSufficiency_" + str(perc)

			self.init_model()
			sigmoidZ, auc = self.run_model()

			aucs.append(auc)

			count += 1

		fp = open(sufficiencyLogFile, "w")
		fp.write(",".join([str(x) for x in aucs]))
		fp.close()

	def cross_validation(self):
		# assume convolutional network
		from sklearn.model_selection import StratifiedKFold

		XvalLogFile = self.params['tag'] + "_KfoldXval.log"
	
		dat = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, True, normalize=False).load_data()

		num_train = dat['X_train'].shape[0]
		num_dev = dat['X_dev'].shape[0]

		all_X = np.concatenate((dat['X_train'], dat['X_dev']), axis=0)
		all_Y = np.concatenate((dat['Y_train'], dat['Y_dev']), axis=0)

		seed = 2

		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)	
		baseTag = self.params['tag']
	
		foldCount = 0

		aucs = list()

		for train_idx, dev_idx in kfold.split(all_X, all_Y):
			X_train = all_X[train_idx,:,:,:]
			Y_train = all_Y[train_idx,:]
			
			normMean = np.mean(X_train, axis=0)
			normStd = np.std(X_train,axis=0) + 0.000000001
			
			for i in range(X_train.shape[0]):
				X_train[i,:,:] = np.divide(X_train[i,:,:] - normMean, normStd)

			X_dev = all_X[dev_idx,:,:,:]
			Y_dev = all_Y[dev_idx,:]
	
			X_dev = np.divide(X_dev - normMean, normStd)

			self.data = {'X_train' : X_train, 'Y_train' : Y_train, 'X_dev' : X_dev, 'Y_dev' : Y_dev}
			self.params['tag'] = baseTag + "_KfoldXval_" + str(foldCount)

			self.init_model()
			sigmoidZ, auc, model_loc = self.run_model()
			aucs.append(auc)
			#def init_interpret(self, model, train_mean, train_std, tag, minibatch = 32, thresh = 0.5):
			self.init_interpret(model_loc, normMean, normStd, self.params['tag'])
			self.run_interpret()


			foldCount += 1

		print("KfoldXvals aucs:")
		print(aucs)

		fp = open(XvalLogFile, "w")
		fp.write(",".join([str(x) for x in aucs]))
		fp.close()

	def load(self):
		if self.params['architecture'].startswith("modelconv"):
			self.data = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, True).load_data()	
		else:
			self.data = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_dev'], self.params['Y_dev'], False, False).load_data()	
	
	def init_model(self):
		if self.params['architecture'].startswith("modelconv"):
			self.model = ModelKeras(self.data, self.params['architecture'], 
				self.params['tag'], 
				learning_rate = self.params['learning_rate'],
				weight_decay = self.params['weight_decay'], 
				num_epochs = self.params['num_epochs'], 
				minibatch_size = self.params['minibatch_size'], 
				print_cost = self.params['print_cost'])	
		else:
			self.model = Model(self.data, self.params['architecture'], 
				self.params['tag'], 
				learning_rate = self.params['learning_rate'],
				weight_decay = self.params['weight_decay'], 
				num_epochs = self.params['num_epochs'], 
				minibatch_size = self.params['minibatch_size'], 
				print_cost = self.params['print_cost'])	

	def run_model(self):
		return self.model.run()
