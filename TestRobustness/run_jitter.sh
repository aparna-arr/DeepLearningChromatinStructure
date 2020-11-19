#!/bin/bash

module load python/3.6.1

abdaTrain="/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/train_5.23.18_unbalanced_unaugmented_xyz.txt"
abdaTest="/oak/stanford/groups/aboettig/Aparna/NNproject/clean_data/test_5.23.18_unbalanced_unaugmented_xyz.txt"

#python3 jitter_xyz.py $abdaTrain 10 0.25 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 10 0.5 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 10 0.75 train_5.23.18
#
#python3 jitter_xyz.py $abdaTrain 20 0.25 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 20 0.5 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 20 0.75 train_5.23.18
#
#python3 jitter_xyz.py $abdaTrain 40 0.25 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 40 0.5 train_5.23.18
#python3 jitter_xyz.py $abdaTrain 40 0.75 train_5.23.18
#
#python3 jitter_xyz.py $abdaTest 10 0.25 test_5.23.18
#python3 jitter_xyz.py $abdaTest 10 0.5 test_5.23.18
#python3 jitter_xyz.py $abdaTest 10 0.75 test_5.23.18
#
#python3 jitter_xyz.py $abdaTest 20 0.25 test_5.23.18
#python3 jitter_xyz.py $abdaTest 20 0.5 test_5.23.18
#python3 jitter_xyz.py $abdaTest 20 0.75 test_5.23.18
#
#python3 jitter_xyz.py $abdaTest 40 0.25 test_5.23.18
#python3 jitter_xyz.py $abdaTest 40 0.5 test_5.23.18
#python3 jitter_xyz.py $abdaTest 40 0.75 test_5.23.18

python3 jitter_xyz.py $abdaTrain 100 1.0 train_5.23.18
python3 jitter_xyz.py $abdaTest 100 1.0 test_5.23.18

python3 jitter_xyz.py $abdaTrain 200 1.0 train_5.23.18
python3 jitter_xyz.py $abdaTest 200 1.0 test_5.23.18

python3 jitter_xyz.py $abdaTrain 400 1.0 train_5.23.18
python3 jitter_xyz.py $abdaTest 400 1.0 test_5.23.18

python3 jitter_xyz.py $abdaTrain 1000 1.0 train_5.23.18
python3 jitter_xyz.py $abdaTest 1000 1.0 test_5.23.18
