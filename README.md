Code for manuscript "Beyond enhancer-promoter contact: leveraging deep learning to connect super-resolution DNA traces to transcription"
All code was written by Aparna R. Rajpurkar.

The computing for this project was performed on the Sherlock cluster. We would like to thank Stanford University and the Stanford Research Computing Center for providing computational resources and support that contributed to these research results.

All deep learning and molecular dynamics simulations were performed on the ```gpu``` partition of Sherlock, on either NVIDIA Tesla P100-PCIE-16GB or Tesla P40 hardware. Additional resources provided by this computing environment include 128 GB RAM, 20 cores, and 200 GB local SSD.
All non-GPU computing was performed on an ```owner``` partition node. This consists of 24 cores, 384GB of memory. 
The operating system in use on the Sherlock cluster is CentOS Linux release 7.6.1810. 

Run time of optimal CNN models is ~3hrs on this hardware, without cross-validation. Run time of all larger models is >12hrs on this hardware. Run time of molecular dynamics simulations is ~24hrs to produce data on this hardware. Post-processing analysis script run times on this hardware vary between seconds to up to an hour.

This software has not been tested on non-Linux platforms and requires the use of a modern GPU.

All dependencies of this software are open source. These include Tensorflow, Keras, Numpy, Matplotlib, Scikit-learn, Scipy. Additional details can be found in the Methods.

To perform replication experiments or test the code, the run script for each test must be modified to point at your specific data directories. For example, to run 10x cross-validation on AbdA labelled data, you need to go to:
https://github.com/aparna-arr/DeepLearningChromatinStructure/blob/master/CNN/KFoldXVal/KfoldXvalAbdA/instance_model_modelconv1_learn_0.00001_wdecay_0.00002_minibatch_32_epochs_500_AbdA_KfoldXval.py

instance_model* defines the "run script" for a neural network model. Here, all modifiable parameters may be accessed. Source code for all optimal CNN models is identical. The only changed parameters between all optimal CNN models is the data path to the appropriate labels.

Here, modify the file paths of:
```X_train
Y_train

X_dev
Y_dev

X_test
Y_test```
to specific data files. Each X* data file must consist of XYZ coordinates, and each Y file must consist of a single column with 1 (ON) or 0 (OFF) labels.

After modifying this file, load cuda on the GPU partition or node you have access to. Load tensorflow and python3. Run the instance_model* run script with:
```python3 instance_model*```

This will then automatically train the network.
