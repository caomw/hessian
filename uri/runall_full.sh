#!/bin/sh

#############################################################################################################
#############################################################################################################

# Baseline

filename="baseline-output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

th ${script_dir_path}/../../src/train_mnist.lua -batchSize 60000 -maxEpoch 200 -full  -currentDir ${script_dir_path}/../../src -gradNormThresh 0.2 -hessianMultiplier 10 -iterMethodDelta 10e-10 -iterationMethod lanczos  -modelpath /models/mnist_small_model.lua 
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "accuracy" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'

### the below is for train_cifar.lua

## for error 
#image_name2="epochPlotError"
#image_name="epochPlotError.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "error rate" -input1 ${script_dir_path}/$filename/trainErr.bin -input2 ${script_dir_path}/$filename/testErr.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/${image_name2}.eps img/${image_name2}.png
#convert img/${image_name2}.png -background white -flatten -alpha off img/${image_name2}.png
#
## for accuracy
#image_name3="epochPlotAccuracy"
#image_name="epochPlotAccuracy.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "accuracy" -input1 ${script_dir_path}/$filename/trainAcc.bin -input2 ${script_dir_path}/$filename/testAcc.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/$image_name3.eps img/hoge.png
#convert img/hoge.png -background white -flatten -alpha off img/$image_name3.png
#
#rm img/*.eps


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)


#############################################################################################################
#############################################################################################################


# hessian-constant

cd ../

filename="hessian-const-output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

th ${script_dir_path}/../../src/train_mnist.lua -batchSize 250 -maxEpoch 1 -full -hessian  -currentDir ${script_dir_path}/../../src -gradNormThresh 0.05 -hessianMultiplier 2 -iterMethodDelta 10e-10 -iterationMethod lanczos  -modelpath /models/reduced-train-mnist-model.lua 
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "accuracy" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'

### the below is for train_cifar.lua

## for error 
#image_name2="epochPlotError"
#image_name="epochPlotError.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "error rate" -input1 ${script_dir_path}/$filename/trainErr.bin -input2 ${script_dir_path}/$filename/testErr.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/${image_name2}.eps img/${image_name2}.png
#convert img/${image_name2}.png -background white -flatten -alpha off img/${image_name2}.png
#
## for accuracy
#image_name3="epochPlotAccuracy"
#image_name="epochPlotAccuracy.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "accuracy" -input1 ${script_dir_path}/$filename/trainAcc.bin -input2 ${script_dir_path}/$filename/testAcc.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/$image_name3.eps img/hoge.png
#convert img/hoge.png -background white -flatten -alpha off img/$image_name3.png
#
#rm img/*.eps


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)


#############################################################################################################
#############################################################################################################


## Linesearch method

cd ../

filename="linesearch-output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename


th ${script_dir_path}/../../src/train_mnist.lua -batchSize 1000 -maxEpoch 200 -full -hessian -lineSearch  -currentDir ${script_dir_path}/../../src -gradNormThresh 0.1 -hessianMultiplier 1 -iterMethodDelta 10e-10 -iterationMethod lanczos  -modelpath /models/mnist_small_model.lua 
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "accuracy" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'

### the below is for train_cifar.lua

## for error 
#image_name2="epochPlotError"
#image_name="epochPlotError.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "error rate" -input1 ${script_dir_path}/$filename/trainErr.bin -input2 ${script_dir_path}/$filename/testErr.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/${image_name2}.eps img/${image_name2}.png
#convert img/${image_name2}.png -background white -flatten -alpha off img/${image_name2}.png
#
## for accuracy
#image_name3="epochPlotAccuracy"
#image_name="epochPlotAccuracy.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "accuracy" -input1 ${script_dir_path}/$filename/trainAcc.bin -input2 ${script_dir_path}/$filename/testAcc.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/$image_name3.eps img/hoge.png
#convert img/hoge.png -background white -flatten -alpha off img/$image_name3.png
#
#rm img/*.eps


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)

#############################################################################################################
#############################################################################################################


### Newton

cd ../

filename="newton-output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

th ${script_dir_path}/../../src/train_mnist.lua -batchSize 250 -maxEpoch 1 -full -hessian -newton  -currentDir ${script_dir_path}/../../src -gradNormThresh 0.1 -hessianMultiplier 1 -iterMethodDelta 10e-10 -iterationMethod lanczos  -modelpath /models/reduced-train-mnist-model.lua 
# passing the right path to train_mnist.lua using ${script_dir_path} which is the current directory where 
# this runall.sh is located.

mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "accuracy" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'

### the below is for train_cifar.lua

## for error 
#image_name2="epochPlotError"
#image_name="epochPlotError.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "error rate" -input1 ${script_dir_path}/$filename/trainErr.bin -input2 ${script_dir_path}/$filename/testErr.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/${image_name2}.eps img/${image_name2}.png
#convert img/${image_name2}.png -background white -flatten -alpha off img/${image_name2}.png
#
## for accuracy
#image_name3="epochPlotAccuracy"
#image_name="epochPlotAccuracy.eps"
#
#th ${script_dir_path}/../../src/plot_table.lua -epochPlotTensor -xlabel epoch -ylabel "accuracy" -input1 ${script_dir_path}/$filename/trainAcc.bin -input2 ${script_dir_path}/$filename/testAcc.bin  -name ${image_name} --save 'img'
#
#convert -density 150 img/$image_name3.eps img/hoge.png
#convert img/hoge.png -background white -flatten -alpha off img/$image_name3.png
#
#rm img/*.eps


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)



