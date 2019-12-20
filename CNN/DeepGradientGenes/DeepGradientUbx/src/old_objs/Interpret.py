from .Model import *
from .ShapImagePlot import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from matplotlib.colors import Normalize



class MidpointNormalize(Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))
class InterpretDeepGradient:
	def __init__(self, X_test, Y_test, X_train,train_avg, train_std, model_loc, tag, minibatch_size, threshold_sig = 0.5):
		self.X_test = X_test
		self.Y_test = Y_test
		self.X_train = X_train
		self.m = self.X_test.shape[0]
		self.barcode_num = self.X_test.shape[1]
		self.train_avg = train_avg
		self.train_std = train_std
		self.threshold_sig = threshold_sig
		self.tag = tag

		self.model = RestoreModelKeras(self.tag, minibatch_size, model_loc) 

		self.filter_X_test = None
		self.filter_Y_test = None
		self.results = dict()

	def run(self):
		import shap
	
		# find top performers for each class, get indicies (test set)
		# choose 5 top performers each, that were TPs
		# (in this case, for the 2-class model)

		sigmoid_Z, auc = self.model.run(self.X_test, self.Y_test)

		## note: this only looks at what is CALLED as ON, not what is ACTUALLY ON in the test set

		print("sigmoid_Z shape " + str(sigmoid_Z.shape))
		sigmoids_with_idx = [[sigmoid_Z[i,0],i] for i in range(len(sigmoid_Z))]

		sorted_sigmoids = sorted(sigmoids_with_idx, reverse=True)

		print("length sorted sigmoids: " + str(len(sorted_sigmoids)))
		print("length sigmoid_Z " + str(len(sigmoid_Z)))

		print("top 5 sorted_sigmoids " + str(sorted_sigmoids[0:5]))
		print("bottom 5 sorted_sigmoids " + str(sorted_sigmoids[len(sorted_sigmoids)-5:]))

		# how many top performers to get? 
		top_num = 10
	
		sorted_sigmoids = np.array(sorted_sigmoids)
	
		top_pos_idxs = sorted_sigmoids[0:top_num,1] # to get idxs
		top_pos_idxs = top_pos_idxs.astype(int)
		top_neg_idxs = sorted_sigmoids[(len(sigmoid_Z) - top_num):len(sigmoid_Z),1] # to get idxs
		top_neg_idxs = top_neg_idxs.astype(int)

		print("top pos idxs")
		print(top_pos_idxs)
		print("top neg idxs")
		print(top_neg_idxs)

		query_idxs = np.concatenate((top_pos_idxs,top_neg_idxs))

		#query_idxs = query_idxs.astype(int)

		print("length query_idxs: " + str(len(query_idxs)))

		# pass to GradientExplainer with all of the training set X as background
		#Xtrain_list = list()
		#for ex in range(self.X_train.shape[0]):
		#	Xtrain_list.append(self.X_train[ex].tolist())
		#e = shap.GradientExplainer(self.model.return_model(), Xtrain_list)
		actual_model = self.model.return_model()
		print("model inputs are: " + str(actual_model.inputs))
		print("len model inputs are: " + str(len(actual_model.inputs)))
		e = shap.GradientExplainer(actual_model, self.X_train)

		
		# get the shap values
		query_images = self.X_test[query_idxs]
		#query_images = np.squeeze(query_images)
		#query_images = query_images.tolist()
		query_images = [query_images]
		#print("shape query_images: " + str(query_images.shape))
		print("len query_images: " + str(len(query_images)))
		#shap_values = e.shap_values(query_images, ranked_outputs = 2)
		shap_values = e.shap_values(query_images)
		# get the class #'s
		class_names = ["ON" for x in range(top_num)] + ["OFF" for x in range(top_num)]
		class_names = np.array(class_names)
		#class_names = class_names.reshape(class_names.shape[0],1)
		shap_values = np.array(shap_values)
		print("shape class_names = " + str(class_names.shape))
		print("shape shap_values = " + str(shap_values.shape))
		# plot the explanations
		test_images = np.squeeze(self.X_test[query_idxs,:,:,0])
		print("shape test_images: " + str(test_images.shape))	
		shap_values_reshape = np.squeeze(shap_values)
		print("shape shap_values_reshape: " + str(shap_values_reshape.shape))
		image_plot(shap_values_reshape, test_images,class_names,pdf_name="SingleGeneShapImage.pdf")

class Interpret:
	def __init__(self, X_test, Y_test, train_avg, train_std, model_loc, tag, minibatch_size, threshold_sig = 0.5):
		self.X_test = X_test
		self.Y_test = Y_test
		self.m = self.X_test.shape[0]
		self.barcode_num = self.X_test.shape[1]
		self.train_avg = train_avg
		self.train_std = train_std
		self.threshold_sig = threshold_sig
		self.tag = tag

		self.model = RestoreModelKeras(self.tag, minibatch_size, model_loc) 

		self.filter_X_test = None
		self.filter_Y_test = None
		self.results = dict()

	def plot_avg_on_off(self):
		Y_test = self.Y_test.T
		posidx = np.nonzero(Y_test)
		posidx = posidx[0]
		negidx = np.nonzero(1-Y_test)
		negidx = negidx[0]
		
		avg_pos = np.sum(self.X_test[:,posidx],axis=1) / len(posidx)
		avg_neg = np.sum(self.X_test[:,negidx],axis=1) / len(negidx)

		avg_pos = avg_pos.reshape(self.barcode_num,self.barcode_num)
		avg_neg = avg_neg.reshape(self.barcode_num,self.barcode_num)

	
		f = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_pos, cmap = 'seismic', norm=norm)
		plt.colorbar()
		f.savefig("Pos_Avg_Y_test.pdf", bbox_inches = 'tight')

		f = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(avg_neg, cmap = 'seismic', norm=norm)
		plt.colorbar()
		f.savefig("Neg_Avg_Y_test.pdf", bbox_inches = 'tight')

	def filter_data(self):
		sigmoid_Z, auc = self.model.run(self.X_test, self.Y_test)
		sigmoid_Z = sigmoid_Z.T
		Y_test = self.Y_test.T

		pos_idxs = np.nonzero((np.logical_and((sigmoid_Z > self.threshold_sig),Y_test > 0) > 0))

		return pos_idxs[0], auc

	def run_test(self):
		sigmoid_Z, auc = self.model.run(self.X_test, self.Y_test)

		y_pred = np.round(sigmoid_Z)

		precision = precision_score(self.Y_test, y_pred)
		recall = recall_score(self.Y_test, y_pred)
		accuracy = accuracy_score(self.Y_test, y_pred)
		f1 = f1_score(self.Y_test, y_pred)

		print("scores: auc roc [" + str(auc) + "] precision [" + str(precision) + "] recall [" + str(recall) + "] accuracy [" + str(accuracy) + "] f1 [" + str(f1) + "]")

		return y_pred, sigmoid_Z

	def run(self):
		#self.plot_avg_on_off()

		pos_idxs, aucTest = self.filter_data()


		pos_idxs = np.arange(self.m)

		num_pos = len(pos_idxs)
		pos_X_test = self.X_test[pos_idxs,:,:,:]
		
		#pos_X_test = pos_X_test.reshape(self.barcode_num,self.barcode_num,num_pos)
		pos_Y_test = self.Y_test[pos_idxs,:]

		fig=plt.figure()
		bwidths = [1,5,10,20,30]
		colors = ['b', 'g', 'r', 'c', 'm']

		for w in range(len(bwidths)):

			bwidth = bwidths[w]
			blanked_scores = list()
			midpoints = list()
			xvals = np.arange(52)
			yvals = np.zeros((52,1))
			ycounts = np.zeros((52,1))
			for b in range(0,self.barcode_num - (bwidth-1)):
				curr_pos_X_test = pos_X_test.copy()
				curr_pos_X_test[:,b:b+bwidth,:,:] = 0
				curr_pos_X_test[:,:,b:b+bwidth,:] = 0
				#curr_pos_X_test = curr_pos_X_test.reshape(-1,num_pos)
			
				sigmoid_Z, auc = self.model.run(curr_pos_X_test, pos_Y_test)
			
				blanked_scores.append(auc)
				midpoints.append(b+bwidth/2)
				yvals[b:b+bwidth,:] += auc
				ycounts[b:b+bwidth,:] += 1

			#print(blanked_scores)
			#print(len(blanked_scores))
		
			yvals = yvals / ycounts
			plt.plot(xvals,yvals, colors[w], label='Blank %d barcodes' % bwidth)
		plt.xlim([0,52])
		plt.ylim([.45,0.7])
		plt.xlabel('Barcodes')
		plt.ylabel('AUC (ROC)')
		plt.legend(loc='lower right')
		plt.title("AUC (ROC) after blanking barcodes (sliding window)")

		fig.savefig("Blank_Barcodes.pdf")
