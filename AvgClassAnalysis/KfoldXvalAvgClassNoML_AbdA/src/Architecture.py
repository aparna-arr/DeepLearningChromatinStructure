import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import subprocess

from .KerasImports import *

np.random.seed(1)

class GetArchitecture():
	def __init__(self, arch_str):
		self.arch = arch_str

	def get(self):
		if self.arch == "modelregress1":
			return ModelRegress1()
		else:
			print("Error! Model name [" + self.arch + "] does not exist in the factory. Try again.", file=sys.stderr)
			sys.exit(1)
			
	
class Architecture():
	def __init__(self):
		self.name = "INIT"

	def get_str(self):
		return self.name
	
class ModelRegress1(Architecture):
	def __init__(self):
		Architecture.__init__(self)	
		self.name = "MODELREGRESS_1"

	def initialize(self):
		W1 = tf.get_variable("W1", [1,2], initializer=tf.contrib.layers.variance_scaling_initializer())
		b1 = tf.get_variable("b1", [1,1], initializer=tf.zeros_initializer())

		parameters = {
			"W1" : W1,
			"b1" : b1,
		}
		
		return parameters

	def forward_prop(self, X,parameters):
		W1 = parameters['W1']
		b1 = parameters['b1']
		
		Z1 = tf.add(tf.matmul(W1,X), b1)
			
		return Z1
