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
	def __init__(self, X_test, Y_test, X_train,train_avg, train_std, model_loc, tag, minibatch_size, nonorm_X_test, threshold_sig = 0.5):
		self.X_test = X_test
		self.Y_test = Y_test
		self.X_train = X_train
		self.X_test_nonorm = nonorm_X_test
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
		self.most_neg_shap(shap_values_reshape, test_images)
		self.most_pos_shap(shap_values_reshape, test_images)

	def run_avg(self):
		import shap
	
		sigmoid_Z, auc = self.model.run(self.X_test, self.Y_test)
		## note: this only looks at what is CALLED as ON, not what is ACTUALLY ON in the test set
		actual_model = self.model.return_model()
		e = shap.GradientExplainer(actual_model, self.X_train)
		
		# get the shap values
		shap_values = e.shap_values([self.X_test])
		shap_values = np.array(shap_values)
		print("shape shap_values = " + str(shap_values.shape))
		shap_values_reshape = np.squeeze(shap_values)
		print("shape shap_values_reshape: " + str(shap_values_reshape.shape))

		mean_shap = np.mean(shap_values_reshape, axis=0)
		abs_shap = np.amax(np.absolute(shap_values_reshape),axis=0)
		pos_idx_bool = (self.Y_test > 0)

		#print("pos_idx_bool shape " + str(pos_idx_bool.shape))
	
		neg_idx_bool = (self.Y_test < 1)

		pos_query_images = self.X_test[np.squeeze(pos_idx_bool),:,:,:]
		neg_query_images = self.X_test[np.squeeze(neg_idx_bool),:,:,:]

		pos_nonorm_images = self.X_test_nonorm[np.squeeze(pos_idx_bool),:,:,:]
		neg_nonorm_images = self.X_test_nonorm[np.squeeze(neg_idx_bool),:,:,:]

		pos_shap = e.shap_values([pos_query_images])
		neg_shap = e.shap_values([neg_query_images])
		
		pos_shap = np.squeeze(pos_shap)
		neg_shap = np.squeeze(neg_shap)

		print("dumping data to .mat")
		import scipy.io

		scipy.io.savemat(self.tag+'_Pos_SHAP_values.mat', {'pos_SHAP' : pos_shap})
		scipy.io.savemat(self.tag+'_Neg_SHAP_values.mat', {'neg_SHAP' : neg_shap})
		scipy.io.savemat(self.tag+'_Pos_SHAP_distances.mat', {'pos_SHAP_dist' : np.squeeze(pos_nonorm_images)})
		scipy.io.savemat(self.tag+'_Neg_SHAP_distances.mat', {'neg_SHAP_dist' : np.squeeze(neg_nonorm_images)})
		print("done dumping data to .mat")

		pos_mean_shap = np.mean(pos_shap, axis=0)
		neg_mean_shap = np.mean(neg_shap, axis=0)


		# plot everything
		
		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(mean_shap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig(self.tag + "_Avg_shap.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(abs_shap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig(self.tag + "_Abs_Max_shap.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(pos_mean_shap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig(self.tag + "_Pos_Avg_shap.pdf", bbox_inches = 'tight')

		fig = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(neg_mean_shap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		fig.savefig(self.tag + "_Neg_Avg_shap.pdf", bbox_inches = 'tight')

		self.most_neg_shap(neg_shap, neg_nonorm_images)
		self.most_pos_shap(pos_shap, pos_nonorm_images)

	def most_neg_shap(self,shap_values, test_images):
		minShap = np.zeros((52,52))
		minDist = np.zeros((52,52))

		fp_shap = open(self.tag + "_most_neg_SHAP.csv", "w")
		fp_dist = open(self.tag + "_most_neg_SHAP_distance.csv", "w")

		test_images = np.squeeze(test_images)

		## de-normalize image data
		#norm_test_images = np.squeeze(norm_test_images)
		#test_images = np.zeros(norm_test_images.shape)

		#print("train avg")
		#print(self.train_avg)

		#for idx in range(norm_test_images.shape[0]):
			#im = norm_test_images[idx,:,:]	
			#print("norm image")
			#print(im)
			#test_images[idx,:,:] = np.multiply(self.train_std - 0.000000001,im) + self.train_avg
			#print("de-norm image")
			#print(test_images[idx,:,:])

		for i in range(52):
			strOutShap = ""
			strOutDist = ""
			for j in range(52):
				exArray = np.squeeze(shap_values[:,i,j])
				#print("shape exArray: " + str(exArray.shape))
				#print("exArray:")
				#print(exArray)
				minShapValue = np.amin(exArray)

				#print("minShapValue = " + str(minShapValue))
				minIdx = np.where(exArray == minShapValue)

				minIdx = minIdx[0]

				if minIdx.shape[0] > 1:
					minIdx = minIdx[0]

				#print("minIdx = " + str(minIdx))

				minDistValue = np.squeeze(test_images[minIdx,i,j])
				#print("minDistValue = " + str(minDistValue))

				minShap[i,j] = minShapValue
				minDist[i,j] = minDistValue

				if j == 0:
					strOutShap = str(minShapValue) 
					strOutDist = str(minDistValue) 
				else:
					strOutShap = strOutShap + "," + str(minShapValue) 
					strOutDist = strOutDist + "," + str(minDistValue) 

			strOutShap += "\n"
			strOutDist += "\n"
			
			fp_shap.write(strOutShap)
			fp_dist.write(strOutDist)			
		
		fp_shap.close()
		fp_dist.close()

		f = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(minShap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		f.savefig(self.tag + "_most_neg_SHAP.pdf", bbox_inches = 'tight')
							
		f = plt.figure()
		plt.imshow(minDist, cmap = 'Greys_r', vmin=0, vmax = 10)
		plt.colorbar()
		f.savefig(self.tag + "_most_neg_SHAP_distance.pdf", bbox_inches = 'tight')
		

	def most_pos_shap(self,shap_values, test_images):
		maxShap = np.zeros((52,52))
		maxDist = np.zeros((52,52))

		fp_shap = open(self.tag + "_most_pos_SHAP.csv", "w")
		fp_dist = open(self.tag + "_most_pos_SHAP_distance.csv", "w")

		test_images = np.squeeze(test_images)

		## de-normalize image data
		#norm_test_images = np.squeeze(norm_test_images)
		#test_images = np.zeros(norm_test_images.shape)

		#print("train avg")
		#print(self.train_avg)

		#for idx in range(norm_test_images.shape[0]):
			#im = norm_test_images[idx,:,:]	
			#print("norm image")
			#print(im)
			#test_images[idx,:,:] = np.multiply(self.train_std - 0.000000001,im) + self.train_avg
			#print("de-norm image")
			#print(test_images[idx,:,:])

		for i in range(52):
			strOutShap = ""
			strOutDist = ""
			for j in range(52):
				exArray = np.squeeze(shap_values[:,i,j])
				maxShapValue = np.amax(exArray)

				maxIdx = np.where(exArray == maxShapValue)

				maxIdx = maxIdx[0]

				if maxIdx.shape[0] > 1:
					maxIdx = maxIdx[0]


				maxDistValue = np.squeeze(test_images[maxIdx,i,j])

				maxShap[i,j] = maxShapValue
				maxDist[i,j] = maxDistValue

				if j == 0:
					strOutShap = str(maxShapValue) 
					strOutDist = str(maxDistValue) 
				else:
					strOutShap = strOutShap + "," + str(maxShapValue) 
					strOutDist = strOutDist + "," + str(maxDistValue) 

			strOutShap += "\n"
			strOutDist += "\n"
			
			fp_shap.write(strOutShap)
			fp_dist.write(strOutDist)			
		
		fp_shap.close()
		fp_dist.close()

		f = plt.figure()
		norm = MidpointNormalize(midpoint = 0)
		plt.imshow(maxShap, cmap = 'seismic', norm=norm)
		plt.colorbar()
		f.savefig(self.tag + "_most_pos_SHAP.pdf", bbox_inches = 'tight')
							
		f = plt.figure()
		plt.imshow(maxDist, cmap = 'Greys_r', vmin=0, vmax = 10)
		plt.colorbar()
		f.savefig(self.tag + "_most_pos_SHAP_distance.pdf", bbox_inches = 'tight')
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
		#return sigmoid_Z

	def run(self):
		#self.plot_avg_on_off()
		from matplotlib import collections  as mc

		pos_idxs, aucTest = self.filter_data()


		pos_idxs = np.arange(self.m)

		num_pos = len(pos_idxs)
		pos_X_test = self.X_test[pos_idxs,:,:,:]
		
		#pos_X_test = pos_X_test.reshape(self.barcode_num,self.barcode_num,num_pos)
		pos_Y_test = self.Y_test[pos_idxs,:]

		larger_bwidths = [20,30]
		smaller_bwidths = [1,5,10]

		fig=plt.figure()
		#bwidths = [1,5,10,20,30]
		#colors = ['b', 'g', 'r', 'c', 'm']
		colors = ['b', 'g', 'r']

		for w in range(len(smaller_bwidths)):

			bwidth = smaller_bwidths[w]
			blanked_scores = list()
			midpoints = list()

			lineCollec = list()
	
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
			
				lineCollec.append([(b,auc), (b+bwidth,auc)])
			#print(blanked_scores)
			#print(len(blanked_scores))
		
			#yvals = yvals / ycounts
			#plt.plot(xvals,yvals, colors[w], label='Blank %d barcodes' % bwidth)
			lc = mc.LineCollection(lineCollec, colors = colors[w], linewidths=2)
			plt.gca().add_collection(lc)

		plt.xlim([0,52])
		plt.ylim([.45,1.0])
		plt.xlabel('Barcodes')
		plt.ylabel('AUC (ROC)')
		plt.legend(loc='lower right')
		plt.title("AUC (ROC) after blanking barcodes (sliding window)")

		fig.savefig("Blank_Barcodes_Smaller_Windows.pdf")

		
		fig=plt.figure()
		#bwidths = [1,5,10,20,30]
		#colors = ['b', 'g', 'r', 'c', 'm']
		colors = ['c', 'm']

		for w in range(len(larger_bwidths)):

			bwidth = larger_bwidths[w]
			blanked_scores = list()
			midpoints = list()

			lineCollec = list()
	
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
			
				lineCollec.append([(b,auc), (b+bwidth,auc)])
			#print(blanked_scores)
			#print(len(blanked_scores))
		
			#yvals = yvals / ycounts
			#plt.plot(xvals,yvals, colors[w], label='Blank %d barcodes' % bwidth)
			lc = mc.LineCollection(lineCollec, colors = colors[w], linewidths=2)
			plt.gca().add_collection(lc)

		plt.xlim([0,52])
		plt.ylim([.45,1.0])
		plt.xlabel('Barcodes')
		plt.ylabel('AUC (ROC)')
		plt.legend(loc='lower right')
		plt.title("AUC (ROC) after blanking barcodes (sliding window)")

		fig.savefig("Blank_Barcodes_Larger_Windows.pdf")
