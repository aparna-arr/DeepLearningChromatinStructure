#!/bin/bash
#SBATCH --job-name=RadTrain200
#SBATCH --output=RadTrain200.%j.out
#SBATCH --error=RadTrain200.%j.err
#SBATCH -p gpu
#SBATCH --time=12:00:00
#SBATCH --gpus 1
#SBATCH --nodes=1
#SBATCH --mem=32G
module load cuda
module load py-tensorflow/1.12.0_py36

#python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.25.py
#python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.5.py
#python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.75.py

#python3 instance_model_train_5.23.18_JitterRad-100.0_jitterPerc-1.0.py 
python3 instance_model_train_5.23.18_JitterRad-200.0_jitterPerc-1.0.py
#python3 instance_model_train_5.23.18_JitterRad-400.0_jitterPerc-1.0.py
#python3 instance_model_train_5.23.18_JitterRad-1000.0_jitterPerc-1.0.py
