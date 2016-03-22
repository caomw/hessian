
#2016/03/16 Meeting 01

The plot for MNIST : modified algorithm (blue)  v.s. regular gradient descent (green). 
Note: this is based on the reduced dataset (2000 samples). The full dataset contains 60000 samples. 

![](./img/epoch_plot.png)

(the plot can be reproduced by /Users/yutaro/Research/spring2016/Hessian/results/2016-03-17/for-meeting/plot.lua)

#2016/03/21 Meeting 02

## MNIST

1. Negative Eigenvalues v.s. Epoch

The blue dot : accuracy per epoch (I had to use a dot-representation in order to plot accuracy v.s. epoch and the negative eigenvalues v.s. minibatches at the same time.)

The green dot : represents either a), b), c), or d) case (described below)

![](./img/mnist-full-negative-eigen/power_plot.png)

The number of the case which...

- a)The norm of gradient is close to zero: 26728 / 90000 (90000 is the number of total parameter updates) 

- b)The second powermethod is called: 26672 / 90000

- c)The cost function decreases:  12531 / 90000 

- d)The cost function increases: 14141 / 90000

Comment: 

- a) and b) tells us that for full mnist, whenever the norm of gradient is close to zero, the algorithm is actually near a saddle point instead of a local minimum. 

- c) and d) tells us that more than half of the time, the algorithm increases the loss, instead of decreasing it. The stepsize might be too big...? I need to look into this more. (Note: "learningRate*5, which I used for this experiment, was chosen based on the result of parameter experiment below; learningRate * 5 seems the best choice based on the test accuracy.)
    

2. Parameter Experiment 


![](./img/mnist-parameter-experiment/parameter_test.png)

(the plot can be reproduced by /Users/yutaro/Research/spring2016/Hessian/results/2016-03-19/plot_para_experiment.lua (with some modification of the input))

3. Full sample plot  

![](./img/mnist-full-negative-eigen/parameter_test.png)

(I stopped the regular one after 10 epochs because this was a sanity check and it keeps having better accuracy than the hessian mode)

This means that I need to tune the stepsize for full dataset as well. (cannot rely on the info from the reduced dataset (2000 data sample set)

## CIFAR-10

1. Haven't found the right setting for the norm of the gradient to converge. 

(Setting 1 -- 4 uses only 2000 data samples.)
(For all the settings, I used:

> learningRate : 0.001
> batchSize : 10
> hessianMultiplier : 5
)

- Setting 1 : No preprocessing. default model (2 conv and 1 fully connected)

![](./img/cifar-100epoch-2000samples/gradientPlot_preprocess.png)

x-axis: the number of parameter update
y-axis: the norm of gradient


- Setting 2: With preprocessing. default model.

![](./img/cifar-100epoch-2000samples/gradientPlot_preprocess.png)

> plot for test error

![](./img/cifar-100epoch-2000samples/epochPlotError_preprocess.png)

- Setting 3: With preprocessing. model with ReLU.

![](./img/cifar-100epoch-2000samples/gradientPlot_preprocess_relu.png)

> plot for test error

![](./img/cifar-100epoch-2000samples/epochPlotError_preprocess_relu.png)

- Setting 4: With preprocessing. model with ReLU and Dropout.

![](./img/cifar-100epoch-2000samples/gradientPlot_preprocess_relu_drop.png)

> plot for test error

![](./img/cifar-100epoch-2000samples/epochPlotError_preprocess_relu_drop.png)

- Setting 5: Full sample

![](./img/cifar-100epoch-50000samples/gradientPlot-2016-03-20-00:16:07.png)


#2016/03/23 Meeting 03



