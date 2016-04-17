#2016/02/20

-Read https://en.wikipedia.org/wiki/Power_iteration

-Implemented power method in test folder. (powermethod.lua)

#2016/02/21

-'lapp' = a small and focused Lua module which aims to make standard command-line parsing easier and intuitive.

(http://lua-users.org/wiki/LappFramework)

-Made /src/models/cnn_mnist.lua  this is just a model.

-Started developing code 


#2016/02/22

- looks like in MNIST, the default learning rate is 0.05. (source: https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua)

- and batchsize default is 10 for MNIST

- gradients with respect to the inputs of the module will be stored in gradInput, which is the output of updateGradInput(input, gradOutput) 

- When you want to initialize the weights the same way throughout one experiment, if you set something like: 

>torch.manualSeed(123)

then it's all good. because manualSeed(123) defines a sequence of random numbers.

If you do net = nn.Sequential() twice in your code, the initial weights will be different because it's looking the first element in the sequence of random numbers,
but the second call is looking at the second element in the sequence of random numbers 

- It looks like the parameter update happens at updateParameter(lr) in Module:sharedAccUpdateGradParameters in Module:backward()?? 

- In normal train() code as in mnist/cifar tutorials, model:backward() only calculates df/dw. The actual parameter updates happen in optim.sgd(feval, parameters, optimState) according to this: https://github.com/torch/optim/blob/master/sgd.lua

**Today's Tips**

- how to flatten tensors? 

    > torch.norm(cparam[2]:view(cparam[2]:nElement()))

- how to time : 

```
 local tic = torch.tic()
 [some job]
 torch.toc(tic) # this will produce the time difference between tic and now
```

- How to use :add()

> If torch.add(tensor1, tensor2)

```
a:add(b) # accumulates all elements of b into a.
torch.add(y, a, b) puts a + b in y.
y:add(a, b) puts a + b in y.
y = torch.add(a, b) returns a new Tensor.
```

> If torch.add(tensor1, value, tensor2)

```
x:add(value, y) multiply-accumulates values of y into x.
z:add(x, value, y) puts the result of x + value * y in z.
torch.add(x, value, y) returns a new Tensor x + value * y.
torch.add(z, x, value, y) puts the result of x + value * y in z
```

> If torch.add(tensor, value)

```
Add the given value to all elements in the Tensor.
y = torch.add(x, value) returns a new Tensor.
x:add(value) add value to all elements in place.
```

- How to use variations of :add() such as :addmv(), :addr(), :addmm()

```
torch.addmv(vec1, mat, vec2)
Performs a matrix-vector multiplication between mat (2D Tensor) and vec2 (1D Tensor) and add it to vec1

torch.addr(mat, vec1, vec2)
Performs the outer-product between vec1 (1D Tensor) and vec2 (1D Tensor).

torch.addmm(M, mat1, mat2)
Performs a matrix-matrix multiplication between mat1 (2D Tensor) and mat2 (2D Tensor).
```


#2016/2/23

- Checked that gradParameters are updated after model:backward(inputs, df_dx) in terminal but parameters not.

- Found out ipairs is an iterator function for Lua. Usage:

```
for i, ver in ipairs(t) do   # t has to be a table; index starts from 1!!
    print(ver)
end
```



#2016/2/24

***Today's tips:***

- How to use :split()

```
a = torch.LongTensor({4,5,2,1,7}):split(3) 
# will make  a table 
# such that {torch.LongTensor({4,5,2}), torch.LongTensor({1,7})} in a
```


- How to use :select()

```
# if mat is a 2D matrix
mat:select(1, t) # selects t th row of the matrix.
mat:select(2, t) # selects t th column of the matrix. 
```

- Read the source code of confusion matrix.

```
self.valids[t] = self.mat[t][t] / self.mat:select(1,t):sum()
# the last part is summing up the t th row of the matrix. So this calculates the t th class accuracy and put it in self.valids[t].

self.unionvalids[t] = self.mat[t][t] / (self.mat:select(1,t):sum()+self.mat:select(2,t):sum()-self.mat[t][t])
# the last part is summing up the t th row and t th column. 
```

- The stuff that appears when you print confusion matrix is written here : function ConfusionMatrix:__tostring__()


#2016/2/26

- I attempted to create an original training script that integrates everything (mnist, cifar, etc), but it's taking a lot of time, so I'm just going to use the pre-developed script from now to get the result. 

- The original script is train.lua. Pre-developed scripts will be located in mnist_experiment and cifar_experiment folder. 

-There is a difference in how data is stored in an original state:
```
testData = torch.load('test_32x32.t7', 'ascii') # this is loading mnist test data

testData.data:size()
# 10000
# 1
# 32
# 32
#o [torch.LongStorage of size 4]

testDataCifar = torch.load('../cifar-10-batches-t7/test_batch.t7', 'ascii')
testDataCifar.data:size()
# 3072
# 10000
# [torch.LongStorage of size 2]
```

For labels, 
```
testData.labels:size() # mnist
# 10000
# [torch.LongStorage of size 1]

testDataCifar.labels:size() # cifar
# 1
# 10000
# [torch.LongStorage of size 2]
```


***Questions***

-What does momentum do...?


#2016/2/29

- gradParameters : stored as [torch.DoubleTensor of size 12]

- norm_gradParam's x-axis is the number of minibatches so far   
- torch.save("norm_gradParam.bin", norm_gradParam)

- changed the path of mnist dataset in dataset-mnist.lua `mnist.path_dataset = 'mnist.t7'` to `mnist.path_dataset = '../../data/mnist.t7'` 


- Ran an experiment to get the feel of how the norm of gradient (gradParams) change over the course of training. 

- Things to do next: Plot the gradParams.bin!


#2016/3/10

- From the look of the plot (test.png), I set the threshold to switch from SGD to Hessian to be 0.01

- To do next: Incorporate power iteration to this mnist train script. 
              Modify /mnist-experiment/plot_table.lua to have another option cmd so that it will automatically save an image to src/image/ directory. 
              Also move this script to src/ from src/mnist-experiment/
             
#2016/3/12

- Dealing with gnuplot. Saving plots to files (https://github.com/torch/gnuplot/blob/master/doc/file.md)

>    Any of the above plotting utilities can also be used for directly plotting into eps or png files, or pdf files if your gnuplot installation allows. A final gnuplot.plotflush() command ensures that all output is written to the file properly.

- require 'paths'  is important when you want make your life easier; it handles the path stuff when saveing stuff to files

```
local filename = paths.concat(opt.save, 'mnist.net')


# filename is now '/current/path/opt.save/mnist.net)
```

#2016/3/13 

- Change the file directory structure according to http://5hun.github.io/quickguide_ja/

- changed the path of mnist dataset in dataset-mnist.lua from mnist.path_dataset = '../../data/mnist.t7' to mnist.path_dataset = '../data/mnist.t7' because I changed the location of dataset-mnist.lua from src/mnist-experiment/ to src/ (and I deleted mnist-experiment folder according to the above new directory structure)

- changed the path of mnist dataset again, because it seems ./runall.sh can't find the right path for the data. Need to investigate more. 
- (from '../data/mnist.t7' to '/Users/yutaro/Research/spring2016/Hessian/data/mnist.t7'


- To do next: I had to change the path for dataset to absolute path (in train_mnist.lua and dataset-mnist.lua. This should be relative for the future use.
  To do next: Make summerize.sh so that I can get a plot for grad_normal.bin easily. 


- Made runall.sh and summerize.sh in 

- In order to run train_mnist.lua etc,.. I added /Users/yutaro/Research/spring2016/Hessian/src to PATH by export PATH=/Users/yutaro/Research/spring2016/Hessian/src:${PATH}

- Added path_remove, path_append, path_prepend command to .bash_profile

- Needed to install brew install coreutils in order to use "greadlink -f" command in shell script (./runall.sh) This is for getting current directory in shell script.

- dofile(file_name) can do  dofile some_string + 'dataset-mnist.lua' , where file_name = some_string + dataset-mnist.lua

- Done with runall.sh. 

- After all, I deleted mnist-experiment folder. It's now in results/2016-02-29. The same result is replicated by runall.sh in results/2016-03-12/output-2016-03-14-03:16:41


- Plot: Since this is batchsize 10, maxEpoch = 20, so 1 epoch 200 (2000 data samples / 10 = 200) so 20 * 200 = 4000 (the values x-axis)
  So the gradParam is collected every minibatch. 

![plot for norm_gradParam](../results/2016-03-13/output-2016-03-14-05:43:32/img/plot-2016-03-14-05:45:01.png) 


- To do next: Start incorporating powermethod into train-mnist.lua

- I ran train_mnist with norm_gradParam[#norm_gradParam + 1] = minibatch_norm_gradParam instead of norm_gradParam[#norm_gradParam + 1] = minibatch_norm_gradParam/opt.batchSize  [2016/03/14] because it seems that dividing is not necessary...?

#2016-03-14

*Note*
- results/2016-03-14/output-2016-03-14-14:13:56 and output-2016-03-14-14:17:37 tells us that:

-  105506, which is the size of gradParameters, is mini-batchsize-invariant. Because it depends on the network architecture. What's dependent on the mini-batchsize is the norm of gradParams. Indeed, see the next pictures.

![/Users/yutaro/Research/spring2016/Hessian/results/2016-03-14/output-2016-03-14-15:36:33/img/plot-2016-03-14-15:36:43.png](../results/2016-03-14/output-2016-03-14-15:36:33/img/plot-2016-03-14-15:36:43.png) 

- batchsize 1, maxEpoch 1, training samples 2000 (x-axis: 2000 / 1)

![/Users/yutaro/Research/spring2016/Hessian/results/2016-03-14/output-2016-03-14-15:43:19/img/plot-2016-03-14-15:43:25.png](../results/2016-03-14/output-2016-03-14-15:43:19/img/plot-2016-03-14-15:43:25.png)

- batchsize 10, maxEpoch 1, training samples 2000 (x-axis: 2000/10)


need to confirm if norm gradParam needs to be divided by minibatch size. 

-> gradParameters do not need to be divided by minibatch size because it doesn't have to do anything with it. (increasing minibatch just smoothes out the bumpy change per each examples.



***Threshold for mnist***

- Looking at the plot below and results/2016-03-14/output-2016-03-14-20:07:25/log/test.log, I decided to use 0.5; the loss in test.log starts to stay the same after 15 epochs out of 40 epoches, which roughly corresponds to (3000/8000) area in the plot below, where I took the y-axis value for this area. 

- Therefore, the Hessian experiment, what we want to see is the improvement after this point. 

![](../results/2016-03-14/output-2016-03-14-20:07:25/img/plot-2016-03-14-20:10:10.png)


***Implementation Notes for Powermethod incorporation***

- Need to check the parameters / model:parameters() / model:getParameters() things. Where to overwirte, etc. 
- How to Make sure that model:backward is done with the updated parameters (= original parameters + eps) ? 


- How to load model :

```
model:add(dofile('models/'..opt.model..'.lua'):cuda())
```

#2016-03-15

- Did the sanity check for hessianPowermethod.lua. Looks good.  torch.cdiv(Hd, d) will yield the same constant eigenvalue.

- Cost actually decreases! by this much: 
```
cost_before
2.345761179924 
cost_after
2.3405652284622
```
This can be reproduced by runall.sh in results/2016-03-15/output-2016-03-15-19:48:42 with the train-mnist.lua at this moement. (go github if you really need and check the commit history.)

^ This was wrong so I need to edit later.

- How to load weights into a model
```
param_new,gradParam_eps = model:getParameters() 
param_new:copy(parameters) # YOU NEED THIS LINE. 
WRONG: param_new = parameters
```

^ I need to investigate why I need this copy thing.

- while(torch.norm(d_old - d) > 10^e-3) criteria for the power iteration was wrong so I changed it to (normHd-normHd_old) > 10^e-3) 

- If you have an almost similar function, you should combine them together. Otherwise, you have to do double-work, when you need to modify some part of the function.


- I think the reason why we have this (figure below) is because parameters are updated again even after hessian update. (No. It was because I was mixing up global and local variables between in-function.)

![](../results/2016-03-15/output-2016-03-15-21:42:07/img/plot-2016-03-15-21:48:22.png)

- To do: still need to investigate why the zero thing happens. Might stem fromm getParameters() so I'm currently reviewing Storage and Module_.flatten(parameters). (https://github.com/torch/nn/blob/master/Module.lua)

- Probably I should use parameters() instead of getParameters()????? But why does getParameters() thing doesn't work even though model is different? Maybe the getParameters() automatically points to the same memory area? ,that's why it can't call twice? 

  MAYBE the cause was global variable / local variable problem???????  -->> YES THAT'S IT


- Finally the algorithm looks working and the test accuracy actually improves in comparison to the normal mnist training. This will be reproduced by runall.sh in results/2016-03-15/output-2016-03-16-03:14:34   The plot is in ./logs/epochPlot.png. This is the same as /Users/yutaro/Research/spring2016/Hessian/results/2016-03-17/exp-output-2016-03-18-19:11:26/ 

- Comparison: 2016-03-14/output-2016-03-14-20\:07\:25/logs/*

![](../results/2016-03-17/for-meeting/epoch_plot.png)

(the plot can be reproduced by /Users/yutaro/Research/spring2016/Hessian/results/2016-03-17/for-meeting/plot.lua)

>  1 % mean class accuracy (test set)                                                                                                                                               
  2  8.7500e+01                                                                                                                                                                    
  3  9.2200e+01                                                                                                                                                                    
  4  9.3300e+01                                                                                                                                                                    
  5  9.3600e+01                                                                                                                                                                    
  6  9.3300e+01                                                                                                                                                                    
  7  9.3700e+01                                                                                                                                                                    
  8  9.3800e+01                                                                                                                                                                    
  9  9.4100e+01                                                                                                                                                                    
 10  9.4300e+01                                                                                                                                                                    
 11  9.4500e+01                                                                                                                                                                    
 12  9.4800e+01                                                                                                                                                                    
 13  9.5000e+01                                                                                                                                                                    
 14  9.5000e+01                                                                                                                                                                    
 15  9.4900e+01                                                                                                                                                                    
 16  9.5000e+01                                                                                                                                                                    
 17  9.5000e+01                                                                                                                                                                    
 18  9.5100e+01                                                                                                                                                                    
 19  9.5000e+01                                                                                                                                                                    
 20  9.5000e+01                                                                                                                                                                    
 21  9.5000e+01                                                                                                                                                                    
 22  9.5000e+01                                                                                                                                                                    
 23  9.5100e+01                                                                                                                                                                    
 24  9.5100e+01                                                                                                                                                                    
 25  9.5100e+01                                                                                                                                                                    
 26  9.5300e+01                                                                                                                                                                    
 27  9.5000e+01                                                                                                                                                                    
 28  9.5400e+01                                                                                                                                                                    
 29  9.5500e+01                                                                                                                                                                    
 30  9.5300e+01                                                                                                                                                                    
 31  9.5400e+01                                                                                                                                                                    
 32  9.5400e+01                                                                                                                                                                    
 33  9.5100e+01                                                                                                                                                                    
 34  9.5200e+01                                                                                                                                                                    
 35  9.5100e+01                                                                                                                                                                    
 36  9.5300e+01                                                                                                                                                                    
 37  9.5300e+01                                                                                                                                                                    
 38  9.5400e+01                                                                                                                                                                    
 39  9.5400e+01                                                                                                                                                                    
 40  9.5600e+01                                                                                                                                                                    
 41  9.5600e+01 


#2016/03/16

- Extend plot_table.lua so that it also does to plot train/test error per epoch (Maybe I can use optim.Logger so I should look up this too)

- To do so, I installed gnu-sed. ('brew install gnu-sed --with-default-names') This will install sed in /usr/local/bin. 

- Lua table: a = {}; a["key"] = 10; then you get a = { key : 10 } 

- Modify runall.sh script and train_mnist.lua so that runall.sh will record all the parameters that could affect the test error. 

- To do: Do the same experiment on cifar10. 

***Note***

- train_mnist uses float tensor for SGD.

- Both train_mnist.lua and train_cifar.lua are supposed to save the net at every epoch, but I commented out "torch.save(model)"

#2016/03/17

- To do: I should check this: for train_cifar.lua, it seems that the original script loops over each example of a minibatch.
If I just feed the entire minibatch one at a time, the result should be the same...(I think the reason why they normalize gradients is because of this loop)

- Attempted to make a script that takes accuracy.log and error.log (from train_cifar.lua). Turned out formatting is messy so will instead modify the way accLogger records acc and error in train_cifar.lua.

- **To do 1st**: 
- But before that..verify we actually get negative eigenvalue. Done.
- Keep track the proportion of when the descent happens (number of calls to the power method) Done.

> The plot is located at results/2016-03-19/output-2016-03-20-01:38:02

- Save the norm of gradients per mini-batch, eigenvalues, difference in loss before and after the update. Done.
- Also time one run of an experiment to see how much hessian algo slows down the stuff. Done. 

#2016/03/18

- Modify plot_table.lua so that it also produces plot for the above recorded values.

- It seems just adding eigenVectors to parameters (so stepSize = 1) was worse than spteSize = learningRate * 5. Maybe it was too big. Maybe the right value lies in-between 1 and learningRate * 5. (result is in 2016-03-17/output-2016-03-18-20:30:42)

-  parameters:add(-eigenVec2) is worse (just to confirm.)  (output-2016-03-18-21:17:27)

- **To do 2nd**
- Add an option for Raphy (for cuda) to train_mnist.lua and train_cifar.lua 
- Tune the stepsize parameter more (I might want to use Raphy for this?)



- At the end of the day, I need to have runall.sh that will produce:
   - the plot of test error of modified algo v.s. regular algo -> I need to have two runs of train-mnist.lua and put the resulting *.bin to the separate folder,
   and then call plot_table.lua and 


#2016/03/19

> comment

- The size of the gradient doesn't converge. What's going on? Why is the algorithm not finding the right direction? 
- One epoch should take about 21.5 min for full Cifar10 (without CUDA) on Raphy

> experiment setting

```
  seed : 1
  learningRate : 0.001
  batchSize : 10
  hessianMultiplier : 5
  hessian : false
  network : ""
  model : "convnet"
  save : "train_cifar"
  maxIter : 5
  powermethodDelta : 1e-05
  preprocess : false
  gradnormThresh : 0.01
  t0 : 1
  momentum : 0
  modelpath : "/models/train-cifar-model.lua"
  full : false (meaning using only 2000 samples)
  threads : 2
  optimization : "SGD"
  maxEpoch : 100
  weightDecay : 0
  currentDir : "/Users/yutaro/Research/spring2016/Hessian/results/2016-03-19/../../src"
  visualize : false
  time : 43min
```

> gradient plot

![](../results/2016-03-19/cifar-regular-output-2016-03-19-17:21:24/img/plot-2016-03-19-18:04:22.png)


To do: I should also plot train_acc, test_acc, etc for this experiment. (I need to write a plot.lua thing probably)

> accuracy plot

![](../results/2016-03-19/cifar-regular-output-2016-03-19-17:21:24/img/epochPlotAccuracy.png)


> error plot

![](../results/2016-03-19/cifar-regular-output-2016-03-19-17:21:24/img/epochPlotError.png)


###a cifar 2000 experiment with preprocessing, with ReLu and Dropout

> accuracy plot 



- I have no idea why whether or not putting echo $(dirname $(greadlink -f $0)) will result in a different behaviour in results/2016-03-19/cifar-regular-output-2016-03-19-17:21:24/generatePlot.sh. Having stuck on this for an hour 

- generatePlot.sh is located at results/2016-03-19/

- Somehow the png file created by gnuplot.pngfigure(filename) doesn't display legend correctly, I had to use gnuplot.epsfigure(filename) and convert eps file to png file.

> convert -density 100 hoge.eps hoge.png

will do it.

But the background is gray. How to resolve this?

(reference: http://rikedan.blogspot.com/2014/09/epslinux.html, http://bluepost69-tech.hatenablog.com/entry/2015/12/01/200327)

Final script

```
convert -density 150 test.eps hoge.png 
convert hoge.png -background white -flatten -alpha off hoge2.png

```

- I completely misunderstood the flag option of lua command. For true/falth flag, you don't need to put (default --- ) stuff.
```
  -- simple.lua
  local args = require ('lapp') [[
  Various flags and option types
    -p          A simple optional flag, defaults to false
    -q,--quiet  A simple flag with long name
    -o  (string)  A required option with argument
    <input> (default stdin)  Optional input file parameter
  ]]
```

To do: I had to resolve the nested folder issue at /Users/yutaro/Research/spring2016/Hessian/results/2016-03-19/parameterExperient1-2016-03-20-05:34:12

^ I resolved.


#2016/03/21

- made a meeting notebook in for-meeting. (I could put some of the plots in this notebook too; when I have time.)

- to do : improve the plot. Especially, the negative eigenvalue thing. (in vain. couldn't find it. Ref: http://www.lighting-torch.com/2015/08/24/plotting-with-torch7/3/)

- to do : change the model (reduce the parameter size) 

2016-03-21/gradient-minibatch-experiment for cifar10. 

- time_it_took.bin is reliable after output-2016-03-22-01:30:18




#2016/03/22

***Notes for future experiments***

- Next time, I should code up generating README file automatically from runall.sh or something (including parameter info, the purpose of the experiment etc)

> Email to Uri
> So I fixed the bug, and I'm about to start running the newton / line-search / constant * learningRate comparison experiment. 
I finished the baseline experiment, and when I was checking the gradient norm, I have the following choices for threshold:

0.01 : This will give us 808 times of passing the first test (gradient threshold test) over 60000 updates.
0.02 : This will give us 2715 times of passing the first test (gradient threshold test) over 60000 updates. 
0.05 : This will give us 13581 times of passing the first test (gradient threshold test) over 60000 updates.

Given the fact that the first plot that we had (the one where our algorithm performs better than the baseline) had 315 / 8000 ratio of passing the gradient norm test, I'm inclined to test it with 0.02 since 315 / 8000 (can be checked here:  /Users/yutaro/Research/spring2016/Hessian/results/2016-03-15/output-2016-03-16-03:14:34) is roughly the same as 2715 / 60000.

Also  




- To do : I should code up for CUDA mode in train_mnist.lua and cifar.lua

- To do: incorporate the saving function (stepsize) Done. When I restart the experiment, I should first git pull in the Raphy repo. And then delete the hessian*constant part. 




#2016/04/16

- Reduced the image size of MNIST to 10 x 10. The script is resize10x10test.lua and resize10x10.lua located in data/mnist.t7

    - I used require 'graphicsmagick' to do image reducing.
