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
		self.data, self.train_mean, self.train_std = ReadData(self.params['X_train'], self.params['Y_train'], self.params['X_test'], self.params['Y_test'], True, False).load_data()	
	
	def init_interpret(self):
		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['architecture'], self.params['threshold_sigmoid'])

	def run(self):
		self.interpretObj.run()
	def run_test(self):
		return self.interpretObj.run_test()

	def make_plots(self):
		# get training set predictions
		self.interpretObj = Interpret(self.data['X_train'], self.data['Y_train'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['architecture'], self.params['threshold_sigmoid'])				
		
		train_y_pred, train_sigZ = self.run_test()

		# get test set predictions
		self.interpretObj = Interpret(self.data['X_dev'], self.data['Y_dev'], self.train_mean, self.train_std, self.params['model_loc'], self.params['tag'], self.params['architecture'], self.params['threshold_sigmoid'])				
		test_y_pred, test_sigZ = self.run_test()

		print("shape train_sigZ: [" + str(train_sigZ.shape) + "]")
		print("top 5 of y_true:[" + str(self.data['Y_train'].T[:,0]))

		fprTrain, tprTrain, thresholdTrain = metrics.roc_curve(self.data['Y_train'].T[:,0], train_sigZ.T[:,0])
		fprTest, tprTest, thresholdTest = metrics.roc_curve(self.data['Y_dev'].T[:,0], test_sigZ.T[:,0])

		aucTrain = metrics.roc_auc_score(self.data['Y_train'].T[:,0], train_sigZ.T[:,0])
		aucTest = metrics.roc_auc_score(self.data['Y_dev'].T[:,0], test_sigZ.T[:,0])

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

	
#		# plot confusion matrix
#		cfTrain = metrics.confusion_matrix(self.data['Y_train'], train_y_pred)	
#
#		fig = plt.figure()
#		plt.imshow(cfTrain,cmap=plt.cm.Blues,interpolation='nearest')
#		plt.colorbar()
#		plt.title('Training Confusion Matrix')
#		plt.xlabel('Predicted')
#		plt.ylabel('Actual')
#		tick_marks = np.arange(2)
#		class_labels = ['0','1']
#		plt.xticks(tick_marks,class_labels)
#		plt.yticks(tick_marks,class_labels)
#		thresh = cfTrain.max() / 2.
#
#		for i,j in itertools.product(range(cfTrain.shape[0]),range(cfTrain.shape[1])):
#			plt.text(j,i,format(cfTrain[i,j],'d'),horizontalalignment='center',color='white' if cfTrain[i,j] > thresh else 'black')
#
#		fig.savefig("ConfusionMat_Train.pdf", bbox_inches='tight')
#			
#		cfTest = metrics.confusion_matrix(self.data['Y_dev'], test_y_pred)	
#
#		fig = plt.figure()
#		plt.imshow(cfTest,cmap=plt.cm.Blues,interpolation='nearest')
#		plt.colorbar()
#		plt.title('Testing Confusion Matrix')
#		plt.xlabel('Predicted')
#		plt.ylabel('Actual')
#		tick_marks = np.arange(2)
#		class_labels = ['0','1']
#		plt.xticks(tick_marks,class_labels)
#		plt.yticks(tick_marks,class_labels)
#		thresh = cfTest.max() / 2.
#
#		for i,j in itertools.product(range(cfTest.shape[0]),range(cfTest.shape[1])):
#			plt.text(j,i,format(cfTest[i,j],'d'),horizontalalignment='center',color='white' if cfTest[i,j] > thresh else 'black')
#
#		fig.savefig("ConfusionMat_Test.pdf", bbox_inches='tight')
#
#		# print average of correctly and incorrectly classified test examples
#		tp_idxs = np.nonzero(np.logical_and(test_y_pred, self.data['Y_dev']))
#		tn_idxs = np.nonzero(np.logical_and((1-test_y_pred), (1-self.data['Y_dev'])))
#		fn_idxs = np.nonzero(np.logical_and((1-test_y_pred), self.data['Y_dev']))
#		fp_idxs = np.nonzero(np.logical_and(test_y_pred, (1-self.data['Y_dev'])))
#		
#		tp_idxs = tp_idxs[0]
#		tn_idxs = tn_idxs[0]
#		fp_idxs = fp_idxs[0]
#		fn_idxs = fn_idxs[0]
#		
#		avg_tp = np.sum(self.data['X_dev'][tp_idxs,:,:],axis=0) / len(tp_idxs)
#		avg_tn = np.sum(self.data['X_dev'][tn_idxs,:,:],axis=0) / len(tn_idxs)
#		avg_fn = np.sum(self.data['X_dev'][fn_idxs,:,:],axis=0) / len(fn_idxs)
#		avg_fp = np.sum(self.data['X_dev'][fp_idxs,:,:],axis=0) / len(fp_idxs)
#
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_tp[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_TP_Y_test.pdf", bbox_inches = 'tight')
#
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_tn[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_TN_Y_test.pdf", bbox_inches = 'tight')
#
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_fp[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_FP_Y_test.pdf", bbox_inches = 'tight')
#	
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_fn[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_FN_Y_test.pdf", bbox_inches = 'tight')
#
#		# plot average pos and neg examples train set
#		pos_idxs = np.nonzero(self.data['Y_train'])
#		neg_idxs = np.nonzero((1-self.data['Y_train']))
#		
#		pos_idxs = pos_idxs[0]
#		neg_idxs = neg_idxs[0]
#		
#		avg_pos = np.sum(self.data['X_train'][pos_idxs,:,:],axis=0) / len(pos_idxs)
#		avg_neg = np.sum(self.data['X_train'][neg_idxs,:,:],axis=0) / len(neg_idxs)
#
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_pos[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_Pos_Y_train.pdf", bbox_inches = 'tight')
#	
#		fig = plt.figure()
#		norm = MidpointNormalize(midpoint = 0)
#		plt.imshow(avg_neg[:,:,0], cmap = 'seismic', norm=norm)
#		plt.colorbar()
#		fig.savefig("Avg_Neg_Y_train.pdf", bbox_inches = 'tight')
#
		

class ModelDriver():
	def __init__(self, params):
		self.params = params
		self.data = None
		self.model = None

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
