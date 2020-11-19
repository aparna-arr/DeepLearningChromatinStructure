#!/bin/bash
#SBATCH --job-name=RFRad10Train
#SBATCH --output=RFRad10Train.%j.out
#SBATCH --error=RFRad10Train.%j.err
#SBATCH -p aboettig
#SBATCH --time=01:00:00

module load python/3.6.1

python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.25.py
python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.5.py
python3 instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.75.py
