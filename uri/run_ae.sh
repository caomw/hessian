# MNIST big AE (GPU)


filename="mnist_AE_output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

th ${script_dir_path}/../../src/train_mnist_AE_cuda.lua -batchSize 200 -coefL2 0.0 -learningRate .05 -batchSizeHessian 1000 -maxEpoch 4000  -maxEpochHessian 50 -full  -hessian -lineSearch -currentDir ${script_dir_path}/../../src -gradNormThresh 3.0 -hessianMultiplier 10 -iterMethodDelta 10e-10 -iterationMethod power  -modelpath /models/mnist_big_AE_model.lua 


mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "MSE" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)



#############################################################################################################
#############################################################################################################
cd ../

# MNIST small AE (CPU)


filename="mnist_AE_output-`date \"+%F-%T"`" # will output something like 2016-03-13-17:35:11

mkdir $filename

script_dir_path=$(dirname $(readlink -f $0))

cd $filename

th ${script_dir_path}/../../src/train_mnist_AE.lua -batchSize 100 -clipGrad true -coefL2 0e-4 -learningRate .01 -batchSizeHessian 1000 -maxEpoch 3000  -maxEpochHessian 50 -full  -hessian -lineSearch -currentDir ${script_dir_path}/../../src -gradNormThresh 2.0 -hessianMultiplier 10 -iterMethodDelta 10e-10 -iterationMethod lanczos  -modelpath /models/mnist_small_AE_model.lua 


mkdir img

image_name="gradientPlot.png" 

th ${script_dir_path}/../../src/plot_table.lua -xlabel "number of updates" -ylabel "the norm of gradient"  -input1 norm_gradParam.bin  -name ${image_name} --save 'img'   
# plot the norm_gradParam.bin and name it as plot-the-date-time-it-was-created.png. Saved in img folder.


### the below is for train_mnist.lua
sed -e "1c\test_acc" ${script_dir_path}/$filename/logs/test.log > ${script_dir_path}/$filename/logs/test.csv 
sed -e "1c\train_acc" ${script_dir_path}/$filename/logs/train.log > ${script_dir_path}/$filename/logs/train.csv

image_name="epochPlot.png"

th ${script_dir_path}/../../src/plot_table.lua -epochPlot -xlabel "epoch" -ylabel "MSE" -input1 ${script_dir_path}/$filename/logs/train.csv -input2 ${script_dir_path}/$filename/logs/test.csv  -name ${image_name} --save 'img'


# cleaning up
cp ${script_dir_path}/$(basename $0) ${script_dir_path}/$(basename $0)_

mv ${script_dir_path}/$(basename $0)_ ${script_dir_path}/$filename/$(basename $0)


