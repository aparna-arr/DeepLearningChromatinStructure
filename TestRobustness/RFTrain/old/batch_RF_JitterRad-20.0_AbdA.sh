#!/bin/bash
#SBATCH --job-name=RFRad20Train
#SBATCH --output=RFRad20Train.%j.out
#SBATCH --error=RFRad20Train.%j.err
#SBATCH -p aboettig
#SBATCH --time=01:00:00

module load python/3.6.1

python3 instance_model_train_5.23.18_JitterRad-20.0_jitterPerc-0.25.py
python3 instance_model_train_5.23.18_JitterRad-20.0_jitterPerc-0.5.py
python3 instance_model_train_5.23.18_JitterRad-20.0_jitterPerc-0.75.py
