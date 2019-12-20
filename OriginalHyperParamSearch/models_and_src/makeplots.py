from src.ModelDriver import *

## This model run script runs the object method that generates all figure plots for the poster/paper

X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_rna_2.txt"

X_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/test_5.23.18_unbalanced_unaugmented_xyz.txt"
Y_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/test_5.23.18_unbalanced_unaugmented_rna_2.txt"

minibatch_size = 32
specific_info = "INTERPRET"
model_loc = "/home/users/arrajpur/NNproject/code/final_code/save/modelconv1-0_learn_1e-05_weight_decay_5e-06_minibatch_32_epochs_500_hyperparam_search_MODEL_CONV_1_20190606-155050/modelconv1-0_learn_1e-05_weight_decay_5e-06_minibatch_32_epochs_500_hyperparam_search_MODEL_CONV_1_20190606-155050"
threshold_sigmoid = 0.5

tag = "BEST_MODEL" + str(minibatch_size) + "-" + "_threshold_sigmoid_" + str(threshold_sigmoid) + "_" + specific_info

PARAMETERS = {
	'X_train' : X_train,
	'Y_train' : Y_train,
	'X_test' : X_test,
	'Y_test' : Y_test,
	'minibatch_size' : minibatch_size,
	'model_loc' : model_loc,
	'threshold_sigmoid' : threshold_sigmoid,
	'tag' : tag,
	'print_cost' : True	
}

modelDriver = ModelDriverInterpret(PARAMETERS)
modelDriver.load()
#modelDriver.make_plots()
modelDriver.make_other_gene_plots()
