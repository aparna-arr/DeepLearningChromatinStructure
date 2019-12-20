from src.ModelDriver import *

X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/train_MultiNoCoop_S45_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/train_MultiNoCoop_S45_rna.txt"

X_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/test_MultiNoCoop_S45_xyz.txt"
Y_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/test_MultiNoCoop_S45_rna.txt"

minibatch_size = 32
specific_info = "MultiNoCoop_S45_Noise05"
model_loc = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/save/MultiNoCoop_S45_MODEL_CONV_1_20191116-001321/MultiNoCoop_S45_MODEL_CONV_1_20191116-001321"
threshold_sigmoid = 0.5

#tag = "BEST_MODEL" + str(minibatch_size) + "-" + "_threshold_sigmoid_" + str(threshold_sigmoid) + "_" + specific_info
tag = specific_info

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

# avg, var, confusion matricies
#modelDriver = ModelDriverInterpret(PARAMETERS)
#modelDriver.load()
#modelDriver.make_plots()

# barcode blanking
#modelDriver = ModelDriverInterpret(PARAMETERS)
#modelDriver.load()
#modelDriver.init_interpret()
#modelDriver.run()

# SHAP
modelDriver = ModelDriverInterpretDeepGradient(PARAMETERS)
modelDriver.load()
modelDriver.init_interpret()
#modelDriver.run()
modelDriver.run_avg()

