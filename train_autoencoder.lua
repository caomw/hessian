require 'torch'
require 'nn'
require 'optim'


----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Autoencoder Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-maxEpoch', 5, 'maximum number of epochs to run')
cmd:option('-currentDir', 'foo', 'current directory where this script is executed')
cmd:option('-gradNormThresh', 0.5, 'threshold of grad norm to switch from gradient descent to hessian')
cmd:option('-hessianMultiplier', 5, 'will determine stepsize used for hessian mode. Stepsize = opt.learningRate * opt.hessianMultiplier')
cmd:option('-iterMethodDelta', 10e-10, 'threshold to stop iteration method; will keep running until norm(Av - lambda v)<delta or until max number of iterations is exceeded')
cmd:option('-preprocess', false, 'preprocess training and test data; necessary if you need more than 60/70% test accuracy.')
cmd:option('-hessian', false, 'turn on hessian mode')
cmd:option('-modelpath', '/models/train-cifar-model.lua', 'path to the model used in hessian mode; must be the same as the model used in normal training')
cmd:option('-plot', false, 'turn on plotting while training')
cmd:option('-newton',false, 'turn on newton-like stepsize')
cmd:option('-lineSearch',false, 'turn on lineSearch')
cmd:text()
opt = cmd:parse(arg)
                                                                                                                                                                                   
torch.save("parameter_info.bin", opt)

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()
   model:add(dofile(opt.currentDir .. opt.modelpath))
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnist> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()



