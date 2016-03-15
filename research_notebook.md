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
