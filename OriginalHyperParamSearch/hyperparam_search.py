import sys
import numpy as np

## This code generates instance model files to run
## various combinations of hyperparameters
## for the neural net
## see examples of outputs in models_and_src directory

########## SET GLOBAL VARIABLES HERE ##########
LEARN_OPTS = np.array([0.0001, 0.00005, 0.00001])
WDECAY_OPTS = np.array([0, 0.5, 1, 2, 5])
EPOCHS_LONG = 5000
EPOCHS_MED = 1000
EPOCHS_SHORT = 500
MINIBATCH_OPTS = np.array([16, 32, 64])
MODEL_OPTS = np.array(["model1", "model2", "model7", "modelconv1", "modelconv2"])
filename_start = "instance_model_"
NUM_MODELS = 5

# train and dev files
X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"

X_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_rna_2.txt"

# set seed
np.random.seed(1)
########## END GLOBAL VARIABLES ############

# write parts of run script that are in common for all iterations
run_script_start = """#!/share/software/user/open/python/3.6.1/bin/python3
from src.ModelDriver import *
## MODIFY THESE PARAMS FOR SPECIFIC RUN ###

X_train = """ + "\"" + X_train + "\"" + """
Y_train = """ + "\"" + Y_train + "\"" + """
X_dev = """ + "\"" + X_dev + "\"" + """
Y_dev = """ + "\"" + Y_dev + "\"" + """

version = """ + "0" + """
specific_info = """ + "\"" + "hyperparam_search" + "\"" + """
"""

run_script_end = """## END OF PARAMS TO MODIFY ##

PARAMETERS = {
\t"X_train" : X_train,
\t"Y_train" : Y_train,
\t"X_dev" : X_dev,
\t"Y_dev" : Y_dev,
\t"architecture" : architecture,
\t"learning_rate" : learning_rate,
\t"weight_decay" : weight_decay,
\t"num_epochs" : num_epochs,
\t"minibatch_size" : minibatch_size,
\t"tag" : tag,
\t"print_cost" : True
}

modelDriver = ModelDriver(PARAMETERS)
modelDriver.load()
modelDriver.init_model()
out = modelDriver.run_model()
"""

# pick random values for tunable hyperparamaters and add to run script
for i in range(NUM_MODELS):
	learn = np.random.choice(LEARN_OPTS)
	wdecay = np.random.choice(WDECAY_OPTS) * learn
	minibatch = np.random.choice(MINIBATCH_OPTS)
	model = np.random.choice(MODEL_OPTS)

	# if running a conv model, all that is needed is 1000 epochs
	# if learning rate =~ 0.0001, only 5000 epochs needed
	if model.startswith("modelconv"):
		epochs = EPOCHS_SHORT
	elif learn <= 0.0001:
		epochs = EPOCHS_MED
	else:
		epochs = EPOCHS_LONG

	# make a filename 
	curr_file = filename_start + model + "_learn_" + str(learn) + "_wdecay_" + str(wdecay) + "_minibatch_" + str(minibatch) + "_epochs_" + str(epochs) + "_abdA" 

	# make run script
	curr_run = run_script_start + """architecture = """ + "\"" + model + "\"" + """
learning_rate = """ + str(learn) + """
weight_decay = """ + str(wdecay) + """
minibatch_size = """ + str(minibatch) + """
num_epochs = """ + str(epochs) + """
tag = architecture + "-" + str(version) + "_learn_" + str(learning_rate) + "_weight_decay_" + str(weight_decay) + "_minibatch_" + str(minibatch_size) + "_epochs_" + str(num_epochs) + "_" + specific_info
""" + run_script_end

	# print run script to file
	fh = open(curr_file + ".py", "w")
	fh.write(curr_run)
	fh.close()
	
	# because I'm running on a cluster system, write a batch file
	batch_file = "batch_" + curr_file + ".sh"
	batch_run = """#!/bin/bash
#SBATCH --job-name=HPS_""" + curr_file + """
#SBATCH --output=HPS_""" + curr_file + """.%j.out
#SBATCH --error=HPS_""" + curr_file + """.%j.err
#SBATCH -p gpu
#SBATCH --time=1-00:00:00
#SBATCH --gres gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arrajpur@stanford.edu
module load python/3.6.1
module load py-tensorflow/1.12.0_py36

""" + curr_file + ".py"

	fh_b = open(batch_file, "w")	
	fh_b.write(batch_run)
	fh_b.close() 
