----------------------------------------------------------------------
-- This script shows how to train different models on the MNIST 
-- dataset, using multiple optimization techniques (SGD, LBFGS)
--
-- This script demonstrates a classical example of training 
-- well-known models (convnet, MLP, logistic regression)
-- on a 10-class classification problem. 
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'cunn'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   -m,--model         (default "convnet")   type of model tor train: convnet | mlp | linear
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.05)        learning rate, for SGD only
   -b,--batchSize     (default 100)          batch size
   --batchSizeHessian     (default 60000)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 1e-1)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   -e,--maxEpoch      (default 50)          maximum number of epochs to run
   --maxEpochHessian      (default 50)          maximum number of epochs to run
   -c,--currentDir    (default "foo")          current directory that is executed this script
   -i, --iterationMethod (default "power")  eigenvalue iteration method (Power or Lanczos)
   -g,--gradNormThresh (default 0.1)        threshold of grad norm to switch from gradient descent to hessian
   -h,--hessianMultiplier   (default 5)     will determine stepsize used for hessian mode. Stepsize = opt.learningRate * opt.hessianMultiplier
   --iterMethodDelta (default 10e-10)       threshold to stop iteration method; will keep running until norm(Av - lambda v)<delta or until max number of iterations is exceeded
   --hessian                                turn on hessian mode 
   --modelpath        (default "/models/mnist_small_model.lua") path to the model used in hessian mode; must be the same as the model used in normal training
   --newton                                 turn on Newton-like stepsize
   --lineSearch                             turn on lineSearch 
   -- max_grad_norm  (default 1)
   -- clipGrad (default false)
]]
torch.save("parameter_info.bin",opt)

local dataset_filepath =  opt.currentDir .. '/dataset-mnist.lua' 
--print(dataset_filepath)
dofile(dataset_filepath)

local iterationMethods_filepath = opt.currentDir .. '/iterationMethods_AE_cuda.lua'
dofile(iterationMethods_filepath)

local update_filepath = opt.currentDir .. '/update.lua'
dofile(update_filepath)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use doubles, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
   --torch.setdefaulttensortype('torch.DoubleTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'1','2','3','4','5','6','7','8','9','10'}

-- geometry: width and height of input images
geometry = {28,28}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()
   model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
   model:add(dofile(opt.currentDir .. opt.modelpath):cuda())
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequgeometryential()
   model = torch.load(opt.network)
end


-- retrieve parameters and gradients

parameters,gradParameters = model:getParameters()

param,gradParam = model:parameters()

-- verbose 
print('<mnist> using model:')
print(model)
print(parameters:size())

print(param)
