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

```{python}
 local tic = torch.tic()
 [some job]
 torch.toc(tic) # this will produce the time difference between tic and now
```

- How to use :add()

> If torch.add(tensor1, tensor2)

```{python}
a:add(b) # accumulates all elements of b into a.
torch.add(y, a, b) puts a + b in y.
y:add(a, b) puts a + b in y.
y = torch.add(a, b) returns a new Tensor.
```

> If torch.add(tensor1, value, tensor2)

```{python}
x:add(value, y) multiply-accumulates values of y into x.
z:add(x, value, y) puts the result of x + value * y in z.
torch.add(x, value, y) returns a new Tensor x + value * y.
torch.add(z, x, value, y) puts the result of x + value * y in z
```

> If torch.add(tensor, value)

```{python}
Add the given value to all elements in the Tensor.
y = torch.add(x, value) returns a new Tensor.
x:add(value) add value to all elements in place.
```

- How to use variations of :add() such as :addmv(), :addr(), :addmm()

```{python}
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

```{python}
for i, ver in ipairs(t) do   # t has to be a table; index starts from 1!!
    print(ver)
end
```



#2016/2/24

***Today's tips:***

- How to use :split()

```{python}
a = torch.LongTensor({4,5,2,1,7}):split(3) 
# will make  a table 
# such that {torch.LongTensor({4,5,2}), torch.LongTensor({1,7})} in a
```


- How to use :select()

```{python}
# if mat is a 2D matrix
mat:select(1, t) # selects t th row of the matrix.
mat:select(2, t) # selects t th column of the matrix. 
```

- Read the source code of confusion matrix.

```{python}
self.valids[t] = self.mat[t][t] / self.mat:select(1,t):sum()
# the last part is summing up the t th row of the matrix. So this calculates the t th class accuracy and put it in self.valids[t].

self.unionvalids[t] = self.mat[t][t] / (self.mat:select(1,t):sum()+self.mat:select(2,t):sum()-self.mat[t][t])
# the last part is summing up the t th row and t th column. 
```

- The stuff that appears when you print confusion matrix is written here : function ConfusionMatrix:__tostring__()


#2016/2/26

- I attempted to create an original training script that integrates everything (mnist, cifar, etc), but it's taking a lot of time, so I'm just going to use the pre-developed script from now to get the result. 
-
- The original script is train.lua. Pre-developed scripts will be located in mnist_experiment and cifar_experiment folder. 

-There is a difference in how data is stored in an original state:
```{python}
testData = torch.load('test_32x32.t7', 'ascii') # this is loading mnist test data

testData.data:size()
# 10000
# 1
# 32
# 32
# [torch.LongStorage of size 4]

testDataCifar = torch.load('../cifar-10-batches-t7/test_batch.t7', 'ascii')
testDataCifar.data:size()
# 3072
# 10000
# [torch.LongStorage of size 2]
```

For labels, 
```{python}
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
-
- norm_gradParam's x-axis is the number of minibatches so far   
- torch.save("norm_gradParam.bin", norm_gradParam)

- changed the path of mnist dataset in dataset-mnist.lua `mnist.path_dataset = 'mnist.t7'` to `mnist.path_dataset = '../../data/mnist.t7'` 


- Ran an experiment to get the feel of how the norm of gradient (gradParams) change over the course of training. 

- Things to do next: Plot the gradParams.bin!


#2016/3/10

- From the look of the plot (test.png), I set the threshold to switch from SGD to Hessian to be 0.01


