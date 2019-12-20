import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import subprocess
import time
from sklearn.metrics import roc_auc_score
from .KerasImports import *
from keras.callbacks import Callback

from keras.models import load_model

from .Architecture import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(1)

class InternalEvaluation(Callback): 
	def __init__(self, validation_data=(), labelstr = "", filename = "",interval=100):
		super(Callback, self).__init__()
		self.interval = interval
		self.labelstr = labelstr
		self.filename = filename
		self.xval, self.yval = validation_data

	def calc_accuracy(self, ypred):
		correct_prediction = np.equal(np.round(ypred), self.yval)
		accuracy_mean = np.mean(correct_prediction)
		return accuracy_mean

	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.interval == 0:
			y_pred = self.model.predict(self.xval, verbose=0)
			score = roc_auc_score(self.yval, y_pred)
			#acc = self.calc_accuracy(y_pred)
			#rec = self.calc_recall(y_pred)
			#pre = self.calc_precision(y_pred)
			loss = logs.get("loss")
			print("internal evaluation - epoch: {:d} - score {:.6f}".format(epoch,score))

			logfp = open(self.filename, "a")
			logfp.write("Cost after epoch %i: %f\n\n" % (epoch, loss))
			#logfp.write(self.labelstr + " accuracy after epoch %i: %f\n" % (epoch, acc))
			#logfp.write(self.labelstr + " recall after epoch %i: %f\n" % (epoch, rec))
			#logfp.write(self.labelstr + " precision after epoch %i: %f\n" % (epoch, pre))
			logfp.write(self.labelstr + " AUC (ROC) after epoch %i: %f\n" % (epoch, score))
			logfp.close()

class ModelKeras():
	def __init__(self, data, architecture, tag, learning_rate = 0.0001, weight_decay = 0.000001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
		self.tag = tag
		self.X_train = data['X_train']
		self.Y_train = data['Y_train']
		self.X_test = data['X_dev']
		self.Y_test = data['Y_dev']
		self.architecture = GetArchitecture(architecture).get()
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.num_epochs = num_epochs
		self.minibatch_size = minibatch_size
		self.print_cost = True
		(self.m,_,_,_) = self.X_train.shape
		self.n_y = self.Y_train.shape[0]

		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		self.savestr = self.tag + "_" + self.architecture.get_str() + "_" + self.timestr
		self.savedir = 'save/' + self.savestr + '/'
		self.logfile = self.savestr + ".log"
		self.predfile = self.savestr + ".predictions"
		self.costpdf = self.savestr + "-Cost.pdf"

	def compute_cost_weighted(self, Y_true, Y_pred):
		pos_weight=3.44
		cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = Y_pred, targets = Y_true, pos_weight=pos_weight))
		return cost		

	def run(self):
		ops.reset_default_graph()
		
		tf.set_random_seed(1)
		seed = 3
	
		print("X_train shape:" + str(self.X_train.shape))	
		model = self.architecture.initialize((52,52,1))
		ivaltrain = InternalEvaluation(validation_data=(self.X_train, self.Y_train), labelstr = "Train", filename = self.tag + ".log", interval=1)
		ival = InternalEvaluation(validation_data=(self.X_test, self.Y_test), labelstr = "Dev", filename = self.tag + ".log", interval=1)
		tfopt = tf.contrib.opt.AdamWOptimizer(learning_rate = self.learning_rate, weight_decay = 0.00002)

		model.compile(optimizer=tfopt, loss=self.compute_cost_weighted, metrics = ['acc'])
		if not os.path.isdir(self.savedir):
			subprocess.run(['mkdir', '-p', self.savedir])

		model.fit(self.X_train, self.Y_train, epochs=self.num_epochs, batch_size=self.minibatch_size, callbacks=[ival,ivaltrain], verbose=0)

		model.save(self.savedir + self.savestr)

		predictions = model.predict(self.X_test, verbose=0)	
		score = roc_auc_score(self.Y_test, predictions)
			
		K.clear_session()

		return predictions, score

# make sure this is keras-compatible for conv models too
class RestoreModel():
	def __init__(self, architecture, tag, model_loc, print_cost = True):
		self.tag = tag
		self.architecture = GetArchitecture(architecture).get()
		self.model_loc = model_loc

	def create_placeholders(self, n_x, n_y):
		X = tf.placeholder(dtype=tf.float32, shape=[n_x,None],name="X")
		Y = tf.placeholder(dtype=tf.float32, shape=[n_y,None],name="Y")
		return X,Y

	def get_predictions(self,Z):
		return tf.round(tf.nn.sigmoid(tf.transpose(Z)))
	
	def calc_accuracy(self, Z,Y):
		correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(Z)), tf.round(Y))
		accuracy_mean = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy_mean

	def calc_recall(self,Z,Y):
		pred = tf.greater(tf.nn.sigmoid(tf.transpose(Z)), 0.5)
		labels = tf.greater(tf.transpose(Y), 0.5)
		tp = tf.logical_and(pred, labels)
		rec = tf.divide(tf.reduce_sum(tf.cast(tp, tf.float32)), tf.reduce_sum(tf.cast(labels, tf.float32)))

		return rec

	def calc_precision(self,Z,Y):
		pred = tf.greater(tf.nn.sigmoid(tf.transpose(Z)), 0.5)
		labels = tf.greater(tf.transpose(Y), 0.5)
		tp = tf.logical_and(pred, labels)
		precision = tf.divide(tf.reduce_sum(tf.cast(tp, tf.float32)), tf.reduce_sum(tf.cast(pred, tf.float32)))

		return precision

	def run(self, X_test, Y_test):
		ops.reset_default_graph()
		
		tf.set_random_seed(1)
		seed = 3

		(n_x, m) = X_test.shape
		n_y = Y_test.shape[0]

		costs = []
		X,Y = self.create_placeholders(n_x, n_y)
		parameters = self.architecture.initialize()
		last_Z = self.architecture.forward_prop(X,parameters)

		prediction = tf.nn.sigmoid(last_Z)
		_,aucTest = tf.metrics.auc(Y,prediction, summation_method='careful_interpolation')

		init0 = tf.initialize_local_variables()
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as sess:
			sess.run(init)
			sess.run(init0)
			
			saver.restore(sess, self.model_loc)	
			testAUC = sess.run(aucTest, feed_dict={X:X_test, Y:Y_test})
			sigmoid_Z = sess.run(prediction, feed_dict={X:X_test})
			return sigmoid_Z, testAUC		

class RestoreModelKeras():
	def __init__(self, tag, minibatch_size, model_loc, print_cost = True):
		self.tag = tag
		self.minibatch_size = minibatch_size
		self.model_loc = model_loc
	
	def run(self, X_test, Y_test):
		ops.reset_default_graph()
		
		tf.set_random_seed(1)
		seed = 3
	
		ival = InternalEvaluation(validation_data=(X_test, Y_test), labelstr = "Test", filename = self.tag + ".eval.log", interval=1)

		model = load_model(self.model_loc)
		predictions = model.predict(X_test, verbose=0)	
		score = roc_auc_score(Y_test, predictions)
	
		K.clear_session()

		return predictions, score

class Model():
	def __init__(self, data, architecture, tag, learning_rate = 0.0001, weight_decay = 0.000001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
		self.tag = tag
		self.X_train = data['X_train']
		self.Y_train = data['Y_train']
		self.X_test = data['X_dev']
		self.Y_test = data['Y_dev']
		self.architecture = GetArchitecture(architecture).get()
		self.learning_rate = learning_rate
		self.weight_decay = weight_decay
		self.num_epochs = num_epochs
		self.minibatch_size = minibatch_size
		self.print_cost = True
		(self.n_x, self.m) = self.X_train.shape
		self.n_y = self.Y_train.shape[0]

		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		self.savestr = self.tag + "_" + self.architecture.get_str() + "_" + self.timestr
		self.savedir = 'save/' + self.savestr + '/'
		self.logfile = self.savestr + ".log"
		self.predfile = self.savestr + ".predictions"
		self.costpdf = self.savestr + "-Cost.pdf"

	def create_placeholders(self, n_x, n_y):
		X = tf.placeholder(dtype=tf.float32, shape=[n_x,None],name="X")
		Y = tf.placeholder(dtype=tf.float32, shape=[n_y,None],name="Y")
		return X,Y
	def create_placeholders2(self, n_x, n_y):
		X = tf.placeholder(dtype=tf.float32, shape=[n_x,None],name="X2")
		Y = tf.placeholder(dtype=tf.float32, shape=[n_y,None],name="Y2")
		return X,Y
	def create_placeholders3(self, n_x, n_y):
		X = tf.placeholder(dtype=tf.float32, shape=[n_x,None],name="X3")
		Y = tf.placeholder(dtype=tf.float32, shape=[n_y,None],name="Y3")
		return X,Y
		
	def compute_cost(self,Z,Y):
		logits = tf.transpose(Z)
		labels = tf.transpose(Y)
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
		return cost

	def compute_cost_weighted(self,Z,Y):
		logits = tf.transpose(Z)
		labels = tf.transpose(Y)
		cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits = logits, targets = labels, pos_weight=3.44))
		return cost
	
	def random_mini_batches(self, seed):
		np.random.seed(seed)
		mini_batches = []

		permutation = list(np.random.permutation(self.m))
		shuffled_X = self.X_train[:,permutation]
		shuffled_Y = self.Y_train[:,permutation].reshape((1,self.m))

		num_complete_minibatches = math.floor(self.m/self.minibatch_size)

		for k in range(0,num_complete_minibatches):
			mini_batch_X = shuffled_X[:,k*self.minibatch_size : (k+1) * self.minibatch_size]
			mini_batch_Y = shuffled_Y[:,k*self.minibatch_size : (k+1) * self.minibatch_size]

			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)

		if self.m % self.minibatch_size != 0:
			rounded_down = self.m - self.minibatch_size * math.floor(self.m / self.minibatch_size)
			mini_batch_X = shuffled_X[:, num_complete_minibatches*self.minibatch_size : (num_complete_minibatches*self.minibatch_size + rounded_down)]
			mini_batch_Y = shuffled_Y[:, num_complete_minibatches*self.minibatch_size : (num_complete_minibatches*self.minibatch_size + rounded_down)]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
			
		return mini_batches

	def get_predictions(self,Z):
		return tf.round(tf.nn.sigmoid(tf.transpose(Z)))
	
	def calc_accuracy(self, Z,Y):
		correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(Z)), tf.round(Y))
		accuracy_mean = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy_mean

	def calc_recall(self,Z,Y):
		pred = tf.greater(tf.nn.sigmoid(tf.transpose(Z)), 0.5)
		labels = tf.greater(tf.transpose(Y), 0.5)
		tp = tf.logical_and(pred, labels)
		rec = tf.divide(tf.reduce_sum(tf.cast(tp, tf.float32)), tf.reduce_sum(tf.cast(labels, tf.float32)))

		return rec

	def calc_precision(self,Z,Y):
		pred = tf.greater(tf.nn.sigmoid(tf.transpose(Z)), 0.5)
		labels = tf.greater(tf.transpose(Y), 0.5)
		tp = tf.logical_and(pred, labels)
		precision = tf.divide(tf.reduce_sum(tf.cast(tp, tf.float32)), tf.reduce_sum(tf.cast(pred, tf.float32)))

		return precision

	def plot_cost(self, costs):
		fig = plt.figure()
		plt.plot(np.log(np.squeeze(costs)))
		plt.ylabel('log(cost)')
		plt.xlabel('iterations(per tens)')
		plt.title("Learning rate =" + str(self.learning_rate))
		fig.savefig(self.costpdf, bbox_inches="tight")

	def print_accuracies(self, last_Z, X, Y, epoch_cost, trainAUC, testAUC, trainAUCPR, testAUCPR, epoch=None):
		accuracy = self.calc_accuracy(last_Z,Y)

		recall = self.calc_recall(last_Z,Y)
		precision = self.calc_precision(last_Z,Y)
	
		if epoch != None:

			print("\n")
			print("Cost after epoch %i: %f\n" % (epoch, epoch_cost))

			print("Train accuracy after epoch %i: %f" % (epoch, accuracy.eval({X:self.X_train, Y:self.Y_train})))
			print("Test accuracy after epoch %i: %f" % (epoch, accuracy.eval({X:self.X_test, Y:self.Y_test})))

			print("Train recall after epoch %i: %f" % (epoch,recall.eval({X:self.X_train, Y:self.Y_train})))
			print("Test recall after epoch %i: %f" % (epoch, recall.eval({X:self.X_test, Y:self.Y_test})))

			print("Train precision after epoch %i: %f" % (epoch, precision.eval({X:self.X_train, Y:self.Y_train})))
			print("Test precision after epoch %i: %f" % (epoch, precision.eval({X:self.X_test, Y:self.Y_test})))
			print("Train AUC (ROC) after epoch %i: %f" %(epoch, trainAUC))
			print("Test AUC (ROC) after epoch %i: %f" %(epoch, testAUC))

			print("Train AUC (PR) after epoch %i: %f" %(epoch, trainAUCPR))
			print("Test AUC (PR) after epoch %i: %f" %(epoch, testAUCPR))

			logfp = open(self.logfile, "a")
			logfp.write("\n")
			logfp.write("Cost after epoch %i: %f\n\n" % (epoch, epoch_cost))

			logfp.write("Train accuracy after epoch %i: %f\n" % (epoch, accuracy.eval({X:self.X_train, Y:self.Y_train})))
			logfp.write("Test accuracy after epoch %i: %f\n" % (epoch, accuracy.eval({X:self.X_test, Y:self.Y_test})))

			logfp.write("Train recall after epoch %i: %f\n" % (epoch,recall.eval({X:self.X_train, Y:self.Y_train})))
			logfp.write("Test recall after epoch %i: %f\n" % (epoch, recall.eval({X:self.X_test, Y:self.Y_test})))

			logfp.write("Train precision after epoch %i: %f\n" % (epoch, precision.eval({X:self.X_train, Y:self.Y_train})))
			logfp.write("Test precision after epoch %i: %f\n" % (epoch, precision.eval({X:self.X_test, Y:self.Y_test})))
			logfp.write("Train AUC (ROC) after epoch %i: %f\n" %(epoch, trainAUC))
			logfp.write("Test AUC (ROC) after epoch %i: %f\n" %(epoch, testAUC))

			logfp.write("Train AUC (PR) after epoch %i: %f\n" %(epoch, trainAUCPR))
			logfp.write("Test AUC (PR) after epoch %i: %f\n" %(epoch, testAUCPR))

			logfp.close()

	def run(self):
		ops.reset_default_graph()
		
		tf.set_random_seed(1)
		seed = 3

		#costs = []
		X,Y = self.create_placeholders(self.n_x, self.n_y)
		parameters = self.architecture.initialize()
		last_Z = self.architecture.forward_prop(X,parameters)

		prediction = tf.nn.sigmoid(last_Z)
		#_,aucTrain = tf.metrics.auc(Y,prediction, summation_method='careful_interpolation')
		_,aucTest = tf.metrics.auc(Y,prediction, summation_method='careful_interpolation')

		#_,aucTrainPR = tf.metrics.auc(Y,prediction,curve="PR", summation_method='careful_interpolation')
		#_,aucTestPR = tf.metrics.auc(Y,prediction, curve="PR", summation_method='careful_interpolation')

		cost = self.compute_cost_weighted(last_Z,Y)
		optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate = self.learning_rate, weight_decay = self.weight_decay).minimize(cost)
		init0 = tf.initialize_local_variables()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		#logfp = open(self.logfile, "w")
		#logfp.write(self.timestr + "\n")
		#logfp.close()
		if not os.path.isdir(self.savedir):
			subprocess.run(['mkdir', '-p', self.savedir])
		
		with tf.Session() as sess:
			sess.run(init)
			sess.run(init0)
			for epoch in range(self.num_epochs):
				epoch_cost = 0.
				num_minibatches = int(self.m / self.minibatch_size)
				seed = seed + 1
				minibatches = self.random_mini_batches(seed)
				
				for minibatch in minibatches:
					(minibatch_X, minibatch_Y) = minibatch
					_, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X:minibatch_X, Y:minibatch_Y})
					epoch_cost += minibatch_cost / num_minibatches
				#if self.print_cost and epoch % 100 == 0:
					#saver.save(sess, self.savedir + self.savestr, global_step = epoch)

					#trainAUC = sess.run(aucTrain, feed_dict={X:self.X_train, Y:self.Y_train})
					#testAUC = sess.run(aucTest, feed_dict={X:self.X_test, Y:self.Y_test})
					#trainAUCPR = sess.run(aucTrainPR, feed_dict={X:self.X_train, Y:self.Y_train})
					#testAUCPR = sess.run(aucTestPR, feed_dict={X:self.X_test, Y:self.Y_test})

					#self.print_accuracies(last_Z, X, Y, epoch_cost, trainAUC, testAUC, trainAUCPR, testAUCPR, epoch)
					
		
				#if self.print_cost and epoch % 5 == 0:
					#costs.append(epoch_cost)
	
			saver.save(sess, self.savedir + self.savestr, global_step = epoch)
					
			#parameters = sess.run(parameters)

			#self.plot_cost(costs)
			
			#trainAUC = sess.run(aucTrain, feed_dict={X:self.X_train, Y:self.Y_train})
			testAUC = sess.run(aucTest, feed_dict={X:self.X_test, Y:self.Y_test})
			#trainAUCPR = sess.run(aucTrainPR, feed_dict={X:self.X_train, Y:self.Y_train})
			#testAUCPR = sess.run(aucTestPR, feed_dict={X:self.X_test, Y:self.Y_test})
			#self.print_accuracies(last_Z, X, Y, epoch_cost, trainAUC, trainDevAUC, testAUC, trainAUCPR, trainDevAUCPR, testAUCPR, self.num_epochs)

			predFunc = self.get_predictions(last_Z)
			preds = predFunc.eval({X:self.X_test})
			
			return preds, testAUC
			#np.savetxt(self.predfile, preds, fmt="%i",delimiter=",")

			#return parameters
