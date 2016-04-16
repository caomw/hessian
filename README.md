#hessian
Experiments on computationally efficient algorithm for non-convex optimization

Log:

2016/02/20: 02:00  setting up. test powermethod.

2016/02/21: 01:30  review on torch. did http://torch.ch/docs/five-simple-examples.html

2016/02/22: 05:00  Read the source code of cifar, mnist, cifar(92.4%) to develop my train() function. Read optim. Read oxford pratical4. Watch this https://www.youtube.com/watch?v=NUKp0c4xb8w. Start writing train().   

2016/02/23: 01:30  Read the source code of cifar, mnist, cifar(92.4%). Continue my train()

2016/02/24: 00:30  Read the source code of confusion matrix.  

2016/02/26: 00:15  Finished train.lua 
            00:40  Started provider.lua. Read the source code for dataloading( mnist, cifar10 )
            01:00  Developed data loading script. Investigated data format in mnist, cifar.

2016/02/29: 01:00  Ran an experiment on how the gradient will change for mnist training (2000 size instead of 10000)

2016/03/12: 00:40  Learned plotting function (gnuplot) and paths library (Filename Manipulation Package)
            01:00  Learned folder directory management in computational experiments

2016/03/13: 02:00  Started writing runall.sh file to manage an experiment in one run of a script. 
            03:00  Continued to develop runall.sh. 

2016/03/14: 02:00  Confirmed gradParameters are collected at every mini batch. As the mini-batchsize increases, the norm of gradParameters will be smoother.
                   Developed runall.sh so that it will automatically move itself to the output folder. 
            01:00  Determined the threshold as 0.5
            02:00  Developed HessianPowerMethod.lua 
            02:00  Developed HessianPowerMethod.lua

2016/03/15

07:00 Finished debugging. Finished the algorithm. Test accuracy improved after 20 epochs (out of 40 epochs) in comparison to normal mnist training. 

2016/03/16

02:15 Developed runall.sh so that it also generates epoch plot.
00:30 Modified train_mnist.lua so that it saves all the parameter information.
03:15 Started train_cifar.lua. Finished (mostly).

2016/03/17

01:00 Attempted to make a script that takes accuracy.log and error.log (from train_cifar.lua). Turned out formatting is messy so will change my plan. 
01:00 Modified train_mnist.lua so that it records the size of eigenvalues?

2016/03/18

02:00 Save the norm of gradients per mini-batch, eigenvalues, difference in loss before and after the update. Keep track the proportion of when the descent happens (number of calls to the power method)
00:40 Experiment.


2016/03/19

04:00 Start writing final project report. Read Bengio's paper and Martens' paper.  
04:00 Cifar experiments. Mnist experiments. Plotting.

2016/03/20

06:00 Developed runall.sh (for train_cifar.lua) so that it also incorporates plotting function for error/accuracy. Fixed bugs for ImageMagitck (convert)

06:00 Prepared for meeting. Ran experiments. Organize plots. 

2016/03/21

01:00 Wrote a part of final report. (about today's meeting, and the next experiment I'll perform.)

00:20 Spent 20 min to try to find a better way to visualize the negative eigenvalu v.s. epoch plot. But couldn't find the solution.

00:45 Reduced the model size. Testing out batchsize thing in cifar. Need to test the reduced models on 2000 samples.

00:30 Ran 2000-batchsize experiment on cifar10. Modified train_mnist.lua to incoporate newton-like stepsize tuning. 

00:20 Started experiment on Raphy about compearing Newton and Regular with batchsize 2000 and full model.

00:30 Wrote line search. To do: modifiy helperFunction.lua. See the result of Raphy. 

01:00 Modified helperFunction.lua and thus finished implementing line search. Testing out line search on mnist. Test finished. Preparing plot for meeting.

00:40 Running experiments on Newton, linesearch, reduced model for mnist full.

2016/03/22

00:30 Fixed bug. Running comparison-newton-linesearch-hessianconstant

05:00 Prepared plots for meeting. Running comparison-newton-linesearch-hessianconstant

2016/03/23

03:00 Trying to fix bug(numerical instability) of power method
01:00 Read about Lanczos method
00:30 Implemented more accurate finite difference 
To do: change f(thing) - f(ff) -> f(theta+eps) - f(theta-eps) / 2eps

2016/03/24

01:30 Lanczos method. 
01:25 Implement Lanczos method


2016/03/25

01:00 Ran the experiment. Fixed the bug in powermethod

2016/03/26

00:40 Prepared Plot. Ran another experiment.

2016/03/27

00:40 Prepared plot.


2016/04/04

02:00 Fixed bug. Incorporate Uri's code. Ran a bunch of test experiments.

2016/04/05

00:30 Fixed bug. Incoporate Uri's update.lua.
