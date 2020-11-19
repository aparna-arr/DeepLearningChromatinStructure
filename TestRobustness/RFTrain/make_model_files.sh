#!/bin/bash

#baseFile025="instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.25.py"
#testFile025="test025.tmp"
#
#baseFile05="instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.5.py"
#testFile05="test05.tmp"
#
#baseFile075="instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-0.75.py"
#testFile075="test075.tmp"

baseFile1="instance_model_train_5.23.18_JitterRad-10.0_jitterPerc-1.0.py"
testFile1="test1.tmp"

### JitterRad = 20
#
#cp $baseFile025 $testFile025
#sed -i "s/Rad-10.0/Rad-20.0/g" $testFile025
#mv "$testFile025" "`echo $baseFile025 | sed "s/Rad-10.0/Rad-20.0/g"`"
#
#cp $baseFile05 $testFile05
#sed -i "s/Rad-10.0/Rad-20.0/g" $testFile05
#mv "$testFile05" "`echo $baseFile05 | sed "s/Rad-10.0/Rad-20.0/g"`"
#
#cp $baseFile075 $testFile075
#sed -i "s/Rad-10.0/Rad-20.0/g" $testFile075
#mv "$testFile075" "`echo $baseFile075 | sed "s/Rad-10.0/Rad-20.0/g"`"
### JitterRad = 40
#
#cp $baseFile025 $testFile025
#sed -i "s/Rad-10.0/Rad-40.0/g" $testFile025
#mv "$testFile025" "`echo $baseFile025 | sed "s/Rad-10.0/Rad-40.0/g"`"
#
#cp $baseFile05 $testFile05
#sed -i "s/Rad-10.0/Rad-40.0/g" $testFile05
#mv "$testFile05" "`echo $baseFile05 | sed "s/Rad-10.0/Rad-40.0/g"`"
#
#cp $baseFile075 $testFile075
#sed -i "s/Rad-10.0/Rad-40.0/g" $testFile075
#mv "$testFile075" "`echo $baseFile075 | sed "s/Rad-10.0/Rad-40.0/g"`"

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
