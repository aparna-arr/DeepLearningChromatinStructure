#!/share/software/user/open/python/3.6.1/bin/python3
from src.ModelDriver import *
## MODIFY THESE PARAMS FOR SPECIFIC RUN ###
X_train = "/oak/stanford/groups/aboettig/Aparna/NNreviews/TestRobustness/jitterData/train_5.23.18_JitterRad-40.0_jitterPerc-0.5_xyz.txt"

Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"
X_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_rna_2.txt"

version = 0
specific_info = "RF_AbdA_train_5.23.18_JitterRad-40.0_jitterPerc-0.5"
architecture = "rf"
num_estimators = 900
min_sample_split = 15
max_depth = None
max_leaf_nodes = 3
random_state = 0
class_weight = "balanced"
n_jobs = -1

tag = specific_info
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
modelDriver.run_model()
