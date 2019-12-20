import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import subprocess

from .KerasImports import *

## This module contains all final neural net architectures
## That were tested with the hyperparameter search

np.random.seed(1)

# An object to generate and return specific Architecture instances
class GetArchitecture():
	def __init__(self, arch_str):
		self.arch = arch_str

	def get(self):
		if self.arch == "modelconv1":
			return ModelConv1()
		elif self.arch == "modelconv2":
			return ModelConv2()
		elif self.arch == "model1":
			return Model1()
		elif self.arch == "model2":
			return Model2()
		elif self.arch == "model7":
			return Model7()
		else:
			print("Error! Model name [" + self.arch + "] does not exist in the factory. Try again.", file=sys.stderr)
			sys.exit(1)
			
	
# Base class for the various architectures
class Architecture():
	def __init__(self):
		self.name = "INIT"

	def get_str(self):
		return self.name

## Specific model architectures below here ##

class ModelConv1(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODEL_CONV_1"

	def initialize(self, input_shape):
		X_input = Input(input_shape)
		X = ZeroPadding2D((4,4))(X_input)
	
		X = Conv2D(32,(7,7), strides=(1,1), name="conv0")(X)
		X = BatchNormalization(axis=3, name="bn0")(X)
		X = Activation('relu')(X)
		
		X = MaxPooling2D((2,2), name='max_pool0')(X)
		
		X = Conv2D(32,(7,7), strides=(1,1), name="conv1")(X)
		X = BatchNormalization(axis=3, name="bn1")(X)
		X = Activation('relu')(X)
		
		X = MaxPooling2D((2,2), name='max_pool1')(X)
		
		X = Flatten()(X)
		X = Dense(1, name='fc', activation='sigmoid')(X)
	
		model = Model(inputs = [X_input], outputs = [X], name = 'ConvModel')
	
		return model

class ModelConv2(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODEL_CONV_2"

	def initialize(self, input_shape):
		X_input = Input(input_shape)
		X = ZeroPadding2D((4,4))(X_input)
	
		X = Conv2D(32,(7,7), strides=(1,1), name="conv0")(X)
		#X = BatchNormalization(axis=3, name="bn0")(X)
		X = Activation('relu')(X)
		
		X = MaxPooling2D((2,2), name='max_pool0')(X)
		
		X = Flatten()(X)
		X = Dense(1, name='fc', activation='sigmoid')(X)
	
		model = Model(inputs = [X_input], outputs = [X], name = 'ConvModel')
	
		return model

class Model1(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODEL_1"

	def initialize(self):
		W1 = tf.get_variable("W1", [20,2704], initializer=tf.contrib.layers.variance_scaling_initializer())
		b1 = tf.get_variable("b1", [20,1], initializer=tf.zeros_initializer())
		W2 = tf.get_variable("W2", [10,20], initializer=tf.contrib.layers.variance_scaling_initializer())
		b2 = tf.get_variable("b2", [10,1], initializer=tf.zeros_initializer())
		W3 = tf.get_variable("W3", [1,10], initializer=tf.contrib.layers.variance_scaling_initializer())
		b3 = tf.get_variable("b3", [1,1], initializer=tf.zeros_initializer())

		parameters = {
			"W1" : W1,
			"b1" : b1,
			"W2" : W2,
			"b2" : b2,
			"W3" : W3,
			"b3" : b3
		}
		
		return parameters

	def forward_prop(self, X,parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']
		W3 = parameters['W3']
		b3 = parameters['b3']
		
		Z1 = tf.add(tf.matmul(W1,X), b1)
		A1 = tf.nn.relu(Z1)
		Z2 = tf.add(tf.matmul(W2,A1), b2)
		A2 = tf.nn.relu(Z2)
		Z3 = tf.add(tf.matmul(W3,A2), b3)
			
		return Z3

class Model2(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODEL_2"

	def initialize(self):
		W1 = tf.get_variable("W1", [200,2704], initializer=tf.contrib.layers.variance_scaling_initializer())
		b1 = tf.get_variable("b1", [200,1], initializer=tf.zeros_initializer())
		W2 = tf.get_variable("W2", [100,200], initializer=tf.contrib.layers.variance_scaling_initializer())
		b2 = tf.get_variable("b2", [100,1], initializer=tf.zeros_initializer())
		W3 = tf.get_variable("W3", [1,100], initializer=tf.contrib.layers.variance_scaling_initializer())
		b3 = tf.get_variable("b3", [1,1], initializer=tf.zeros_initializer())

		parameters = {
			"W1" : W1,
			"b1" : b1,
			"W2" : W2,
			"b2" : b2,
			"W3" : W3,
			"b3" : b3
		}
		
		return parameters

	def forward_prop(self, X,parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']
		W3 = parameters['W3']
		b3 = parameters['b3']
		
		Z1 = tf.add(tf.matmul(W1,X), b1)
		A1 = tf.nn.relu(Z1)
		Z2 = tf.add(tf.matmul(W2,A1), b2)
		A2 = tf.nn.relu(Z2)
		Z3 = tf.add(tf.matmul(W3,A2), b3)
			
		return Z3

class Model7(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODEL_7"

	def initialize(self):
		W1 = tf.get_variable("W1", [50,2704], initializer=tf.contrib.layers.variance_scaling_initializer())
		b1 = tf.get_variable("b1", [50,1], initializer=tf.zeros_initializer())
		W2 = tf.get_variable("W2", [25,50], initializer=tf.contrib.layers.variance_scaling_initializer())
		b2 = tf.get_variable("b2", [25,1], initializer=tf.zeros_initializer())
		W3 = tf.get_variable("W3", [10,25], initializer=tf.contrib.layers.variance_scaling_initializer())
		b3 = tf.get_variable("b3", [10,1], initializer=tf.zeros_initializer())
		W4 = tf.get_variable("W4", [10,10], initializer=tf.contrib.layers.variance_scaling_initializer())
		b4 = tf.get_variable("b4", [10,1], initializer=tf.zeros_initializer())
		W5 = tf.get_variable("W5", [1,10], initializer=tf.contrib.layers.variance_scaling_initializer())
		b5 = tf.get_variable("b5", [1,1], initializer=tf.zeros_initializer())

		parameters = {
			"W1" : W1,
			"b1" : b1,
			"W2" : W2,
			"b2" : b2,
			"W3" : W3,
			"b3" : b3,
			"W4" : W4,
			"b4" : b4,
			"W5" : W5,
			"b5" : b5
		}
		
		return parameters

	def forward_prop(self, X,parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		W2 = parameters['W2']
		b2 = parameters['b2']
		W3 = parameters['W3']
		b3 = parameters['b3']
		W4 = parameters['W4']
		b4 = parameters['b4']
		W5 = parameters['W5']
		b5 = parameters['b5']
		
		Z1 = tf.add(tf.matmul(W1,X), b1)
		A1 = tf.nn.relu(Z1)
		Z2 = tf.add(tf.matmul(W2,A1), b2)
		A2 = tf.nn.relu(Z2)
		Z3 = tf.add(tf.matmul(W3,A2), b3)
		A3 = tf.nn.relu(Z3)
		Z4 = tf.add(tf.matmul(W4,A3), b4)
		A4 = tf.nn.relu(Z4)
		Z5 = tf.add(tf.matmul(W5,A4), b5)
			
		return Z5
