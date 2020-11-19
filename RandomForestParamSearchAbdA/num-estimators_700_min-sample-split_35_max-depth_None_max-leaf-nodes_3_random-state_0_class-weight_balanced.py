#!/share/software/user/open/python/3.6.1/bin/python3
from src.ModelDriver import *
## MODIFY THESE PARAMS FOR SPECIFIC RUN ###

X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"
X_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_rna_2.txt"

specific_info = "hyperparam-search-rf"
architecture = "rf"
num_estimators = 700
min_sample_split = 35
max_depth = None
max_leaf_nodes = 3
random_state = 0
class_weight = "balanced"
n_jobs = -1

writestr = "num-estimators_" + str(num_estimators) + "_" +\
        "min-sample-split_" + str(min_sample_split) + "_" +\
        "max-depth_" + str(max_depth) + "_" +\
        "max-leaf-nodes_" + str(max_leaf_nodes) + "_" +\
        "random-state_" + str(random_state) + "_" +\
        "class-weight_" + class_weight
tag = writestr + "_" + specific_info

## END OF PARAMS TO MODIFY ##

PARAMETERS = {
	"X_train" : X_train,
	"Y_train" : Y_train,
	"X_dev" : X_dev,
	"Y_dev" : Y_dev,
	"architecture" : architecture,
	"num_estimators" : num_estimators,
	"min_sample_split" : min_sample_split,
	"max_depth" : max_depth,
	"max_leaf_nodes" : max_leaf_nodes,
	"random_state" : random_state,
	"class_weight" : class_weight,
	"n_jobs" : n_jobs,
	"tag" : tag,
	"print_cost" : True
}

modelDriver = ModelDriver(PARAMETERS)
modelDriver.load()
modelDriver.init_model()
out = modelDriver.run_model()
