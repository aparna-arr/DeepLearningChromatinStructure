from src.ModelDriver import *

X_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/train_MultiCoop_S45_V2_xyz.txt"
Y_train = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/train_MultiCoop_S45_V2_rna.txt"

X_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/test_MultiCoop_S45_V2_xyz.txt"
Y_test = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/process_scripts/test_MultiCoop_S45_V2_rna.txt"

minibatch_size = 32
specific_info = "MultiCoop_S45_V2"
model_loc = "/oak/stanford/groups/aboettig/Aparna/NNproject/RandomPolymerControl/save/MultiCoop_S45_V2_MODEL_CONV_1_20191119-153813/MultiCoop_S45_V2_MODEL_CONV_1_20191119-153813"
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

