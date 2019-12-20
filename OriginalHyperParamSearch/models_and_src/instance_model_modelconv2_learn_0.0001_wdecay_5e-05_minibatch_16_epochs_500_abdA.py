#!/share/software/user/open/python/3.6.1/bin/python3
from src.ModelDriver import *
## MODIFY THESE PARAMS FOR SPECIFIC RUN ###

X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"
X_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_dev = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/dev_5.23.18_unbalanced_unaugmented_rna_2.txt"

version = 0
specific_info = "hyperparam_search"
architecture = "modelconv2"
learning_rate = 0.0001
weight_decay = 5e-05
minibatch_size = 16
num_epochs = 500
tag = architecture + "-" + str(version) + "_learn_" + str(learning_rate) + "_weight_decay_" + str(weight_decay) + "_minibatch_" + str(minibatch_size) + "_epochs_" + str(num_epochs) + "_" + specific_info
## END OF PARAMS TO MODIFY ##

PARAMETERS = {
	"X_train" : X_train,
	"Y_train" : Y_train,
	"X_dev" : X_dev,
	"Y_dev" : Y_dev,
	"architecture" : architecture,
	"learning_rate" : learning_rate,
	"weight_decay" : weight_decay,
	"num_epochs" : num_epochs,
	"minibatch_size" : minibatch_size,
	"tag" : tag,
	"print_cost" : True
}

modelDriver = ModelDriver(PARAMETERS)
modelDriver.load()
modelDriver.init_model()
out = modelDriver.run_model()
