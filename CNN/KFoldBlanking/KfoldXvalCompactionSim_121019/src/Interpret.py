from .Model import *

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
		from matplotlib import collections  as mc
		pos_idxs = np.arange(self.m)

		num_pos = len(pos_idxs)
		pos_X_test = self.X_test[pos_idxs,:,:,:]
		
		#pos_X_test = pos_X_test.reshape(self.barcode_num,self.barcode_num,num_pos)
		pos_Y_test = self.Y_test[pos_idxs,:]

		sigmoid_Z, trashauc = self.model.run(pos_X_test, pos_Y_test)
		refAuc = roc_auc_score(pos_Y_test, sigmoid_Z)
	
		#fig =plt.figure()
		bwidths = [1,5,10,20,30]
		#colors = ['b', 'g', 'r', 'c', 'm']

		fp = open(self.tag + "_blanking_values.txt", "w")
		fp.write(str(refAuc) + "\n")
		for w in range(len(bwidths)):

			bwidth = bwidths[w]
			#blanked_scores = list()
			#midpoints = list()

			#lineCollec = list()

			#xvals = np.arange(52)
			#yvals = np.zeros((52,1))
			#ycounts = np.zeros((52,1))
			for b in range(0,self.barcode_num - (bwidth-1)):
				curr_pos_X_test = pos_X_test.copy()
				curr_pos_X_test[:,b:b+bwidth,:,:] = 0
				curr_pos_X_test[:,:,b:b+bwidth,:] = 0
				#curr_pos_X_test = curr_pos_X_test.reshape(-1,num_pos)
			
				sigmoid_Z, auc = self.model.run(curr_pos_X_test, pos_Y_test)
			
				#blanked_scores.append(auc)
				#midpoints.append(b+bwidth/2)
				#yvals[b:b+bwidth,:] += auc
				#ycounts[b:b+bwidth,:] += 1

				#lineCollec.append([(b,auc), (b+bwidth,auc)])

				writestr = str(bwidth) + "\t" + str(b) + "\t" + str(auc) + "\t" + str(b+bwidth) + "\t" + str(auc) + "\n"
				fp.write(writestr)	
			#print(blanked_scores)
			#print(len(blanked_scores))
		
			#yvals = yvals / ycounts
			#plt.plot(xvals,yvals, colors[w], label='Blank %d barcodes' % bwidth)

			#lc = mc.LineCollection(lineCollec, colors = colors[w], linewidths=2)
			#plt.gca().add_collection(lc)

		fp.close()

		#xvals = np.arange(52)
		#yvals = np.repeat(refAuc,52)
		#plt.plot(xvals,yvals, 'gray', label='Reference AUC')

		#plt.xlim([0,52])
		#plt.ylim([.45,0.7])
		#plt.xlabel('Barcodes')
		#plt.ylabel('AUC (ROC)')
		#plt.legend(loc='lower right')
		#plt.title("AUC (ROC) after blanking barcodes (sliding window)")

		#fig.savefig(self.tag + "_AbdA_Blank_Barcodes.pdf")
