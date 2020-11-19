#!/bin/bash
#SBATCH --job-name=RFRad
#SBATCH --output=RFRad.%j.out
#SBATCH --error=RFRad.%j.err
#SBATCH -p aboettig
#SBATCH --time=1-00:00:00

module load python/3.6.1

python3 instance_model_train_5.23.18_JitterRad-100.0_jitterPerc-1.0.py
python3 instance_model_train_5.23.18_JitterRad-200.0_jitterPerc-1.0.py
python3 instance_model_train_5.23.18_JitterRad-400.0_jitterPerc-1.0.py
python3 instance_model_train_5.23.18_JitterRad-1000.0_jitterPerc-1.0.py
