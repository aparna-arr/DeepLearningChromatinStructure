import sys
import numpy as np

## This code generates instance model files to run
## various combinations of hyperparameters
## for the neural net
## see examples of outputs in models_and_src directory

########## SET GLOBAL VARIABLES HERE ##########
NUM_ESTIMATORS = np.arange(100, 1000, 100)
MIN_SAMPLE_SPLIT = np.arange(5,50,10)
MAX_DEPTH = None
MAX_LEAF_NODES = np.arange(2,4,1)
RANDOM_STATE = 0
N_JOBS = -1

filename_start = "instance_model_"
NUM_MODELS = 150

# train and dev files
X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"

X_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_rna_2.txt"

# set seed
np.random.seed(0)
########## END GLOBAL VARIABLES ############

# write parts of run script that are in common for all iterations
run_script_start = """#!/share/software/user/open/python/3.6.1/bin/python3
from src.ModelDriver import *
## MODIFY THESE PARAMS FOR SPECIFIC RUN ###

X_train = """ + "\"" + X_train + "\"" + """
Y_train = """ + "\"" + Y_train + "\"" + """
X_dev = """ + "\"" + X_dev + "\"" + """
Y_dev = """ + "\"" + Y_dev + "\"" + """

specific_info = """ + "\"" + "hyperparam-search-rf" + "\"" + """
"""

run_script_end = """## END OF PARAMS TO MODIFY ##

PARAMETERS = {
\t"X_train" : X_train,
\t"Y_train" : Y_train,
\t"X_dev" : X_dev,
\t"Y_dev" : Y_dev,
\t"architecture" : architecture,
\t"num_estimators" : num_estimators,
\t"min_sample_split" : min_sample_split,
\t"max_depth" : max_depth,
\t"max_leaf_nodes" : max_leaf_nodes,
\t"random_state" : random_state,
\t"class_weight" : class_weight,
\t"n_jobs" : n_jobs,
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
	architecture = "rf"
	num_estimators = np.random.choice(NUM_ESTIMATORS)
	min_sample_split = np.random.choice(MIN_SAMPLE_SPLIT)
	max_depth = MAX_DEPTH
	max_leaf_nodes = np.random.choice(MAX_LEAF_NODES)
	random_state = RANDOM_STATE
	class_weight = "balanced"
	n_jobs = N_JOBS
	

	# make a filename 
	writestr = "num-estimators_" + str(num_estimators) + "_" +\
        "min-sample-split_" + str(min_sample_split) + "_" +\
        "max-depth_" + str(max_depth) + "_" +\
        "max-leaf-nodes_" + str(max_leaf_nodes) + "_" +\
        "random-state_" + str(random_state) + "_" +\
        "class-weight_" + class_weight
	filename = writestr 

	# make run script
	curr_run = run_script_start + """architecture = """ + "\"" + architecture + "\"" + """
num_estimators = """ + str(num_estimators) + """
min_sample_split = """ + str(min_sample_split) + """
max_depth = """ + str(max_depth) + """
max_leaf_nodes = """ + str(max_leaf_nodes) + """
random_state = """ + str(random_state) + """
class_weight = """ + "\"" + class_weight + "\"" + """
n_jobs = """ + str(n_jobs) + """

writestr = "num-estimators_" + str(num_estimators) + "_" +\\
        "min-sample-split_" + str(min_sample_split) + "_" +\\
        "max-depth_" + str(max_depth) + "_" +\\
        "max-leaf-nodes_" + str(max_leaf_nodes) + "_" +\\
        "random-state_" + str(random_state) + "_" +\\
        "class-weight_" + class_weight
tag = writestr + "_" + specific_info

""" + run_script_end

	# print run script to file
	fh = open(filename + ".py", "w")
	fh.write(curr_run)
	fh.close()
	
	# because I'm running on a cluster system, write a batch file
	batch_file = "batch_" + filename + ".sh"
	batch_run = """#!/bin/bash
#SBATCH --job-name=HPS_""" + filename + """
#SBATCH --output=HPS_""" + filename + """.%j.out
#SBATCH --error=HPS_""" + filename + """.%j.err
#SBATCH -p aboettig
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=20

module load python/3.6.1

python3 """ + filename + ".py"

	fh_b = open(batch_file, "w")	
	fh_b.write(batch_run)
	fh_b.close() 
