
#2016/03/16 Meeting 01

The plot for MNIST : modified algorithm (blue)  v.s. regular gradient descent (green). 
Note: this is based on the reduced dataset (2000 samples). The full dataset contains 60000 samples. 

![](./img/epoch_plot.png)

(the plot can be reproduced by /Users/yutaro/Research/spring2016/Hessian/results/2016-03-17/for-meeting/plot.lua)

#2016/03/21 Meeting 02

## MNIST

1. Parameter Experiment 


![](./img/mnist-parameter-experiment/parameter_test.png)

(the plot can be reproduced by /Users/yutaro/Research/spring2016/Hessian/results/2016-03-19/plot_para_experiment.lua (with some modification of the input))


2. 


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


