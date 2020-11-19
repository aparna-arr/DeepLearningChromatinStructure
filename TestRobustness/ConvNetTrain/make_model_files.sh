#!/bin/bash

baseFile1="instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-1.0.py"
testFile1="test1.tmp"

cp $baseFile1 $testFile1
sed -i "s/Rad-10.0/Rad-100.0/g" $testFile1
mv "$testFile1" "`echo $baseFile1 | sed "s/Rad-10.0/Rad-100.0/g"`"

cp $baseFile1 $testFile1
sed -i "s/Rad-10.0/Rad-200.0/g" $testFile1
mv "$testFile1" "`echo $baseFile1 | sed "s/Rad-10.0/Rad-200.0/g"`"

cp $baseFile1 $testFile1
sed -i "s/Rad-10.0/Rad-400.0/g" $testFile1
mv "$testFile1" "`echo $baseFile1 | sed "s/Rad-10.0/Rad-400.0/g"`"

cp $baseFile1 $testFile1
sed -i "s/Rad-10.0/Rad-1000.0/g" $testFile1
mv "$testFile1" "`echo $baseFile1 | sed "s/Rad-10.0/Rad-1000.0/g"`"
