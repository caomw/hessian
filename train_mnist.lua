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
   -b,--batchSize     (default 10)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 4)           number of threads
   -e,--maxEpoch      (default 50)          maximum number of epochs to run
   -c,--currentDir    (default "foo")          current directory that is executed this script
   -g,--gradnormThresh (default 0.5)        threshold of grad norm to switch from gradient descent to hessian
   -h,--hessianMultiplier   (default 5)     will determine stepsize used for hessian mode. Stepsize = opt.learningRate * opt.hessianMultiplier
   --powermethodDelta (default 10e-6)       threshold to stop powermethod; will keep running until difference between norm_Hd and norm_Hd_old < powermethodDelta
   --hessian                                turn on hessian mode 
   --modelpath        (default "/models/train-train-model.lua") path to the model used in hessian mode; must be the same as the model used in normal training
   --newton                                 turn on Newton-like stepsize
]]

torch.save("parameter_info.bin",opt)

local dataset_filepath =  opt.currentDir .. '/dataset-mnist.lua' 
--print(dataset_filepath)
dofile(dataset_filepath)

local hessian_filepath = opt.currentDir .. '/helperFunctions.lua'
dofile(hessian_filepath)
--local hessian2_filepath = opt.currentDir .. '/test/negativePowermethod.lua'
--dofile(hessian2_filepath)

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- use floats, for SGD
if opt.optimization == 'SGD' then
   torch.setdefaulttensortype('torch.FloatTensor')
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
geometry = {32,32}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()

   if opt.model == 'convnet' then
      ------------------------------------------------------------
      -- convolutional network 
      ------------------------------------------------------------
      -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
      -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
      model:add(nn.Tanh())
      model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer MLP:
      model:add(nn.Reshape(64*2*2))
      model:add(nn.Linear(64*2*2, 200))
      model:add(nn.Tanh())
      model:add(nn.Linear(200, #classes))
      ------------------------------------------------------------

   elseif opt.model == 'mlp' then
      ------------------------------------------------------------
      -- regular 2-layer MLP
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024, 2048))
      model:add(nn.Tanh())
      model:add(nn.Linear(2048,#classes))
      ------------------------------------------------------------

   elseif opt.model == 'linear' then
      ------------------------------------------------------------
      -- simple linear model: logistic regression
      ------------------------------------------------------------
      model:add(nn.Reshape(1024))
      model:add(nn.Linear(1024,#classes))
      ------------------------------------------------------------

   else
      print('Unknown model type')
      cmd:text()
      error()
   end
else
   print('<trainer> reloading previously trained network')
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

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 60000
   nbTestingPatches = 10000
else
   nbTrainingPatches = 2000
   nbTestingPatches = 1000
   print('<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)')
end

-- create training set and normalize
trainData = mnist.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)
norm_gradParam = {}
minibatch_norm_gradParam = 0

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

cost_before_acc = {}
cost_after_acc = {}
eigenTable = {}
eigenTableNeg = {}
powercallRecord = {}

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
         --print("cost in general")
         

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         --minibatch_norm_gradParam = minibatch_norm_gradParam + torch.norm(gradParameters)
         minibatch_norm_gradParam = torch.norm(gradParameters) 

         local clock = os.clock
         function sleep(n)  -- seconds
               local t0 = clock()
               while clock() - t0 <= n do end
         end

         if opt.hessian then
             local flag = 0
         if torch.norm(gradParameters) < opt.gradnormThresh then
             flag = flag + 1
             eigenVec, eigenVal = hessianPowermethod(inputs,targets,parameters:clone(),gradParameters:clone(),opt.powermethodDelta,opt.currentDir,opt.modelpath)
             eigenTable[#eigenTable+1] = eigenVal
             --if eigenVal > 0 then
             --I don't need this condition because eigenVal is always positive (absolute value)
             eigenVec2, eigenVal2 = negativePowermethod(inputs,targets,parameters:clone(),gradParameters:clone(),opt.powermethodDelta,opt.currentDir,eigenVal,opt.modelpath)
             if eigenVal2 > eigenVal then --the Hessian has a negative eigenvalue so we should proceed to this direction
                 flag = flag + 1
                 eigenTableNeg[#eigenTableNeg+1] = eigenVal - eigenVal2
                 cost_before = computeCurrentLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath) 
                 --outputs_before = model:forward(inputs)
                 --cost_before = criterion:forward(outputs, targets)
                 --parameters:copy(parameters + eigenVec2 * stepSize)
                 stepSize = opt.learningRate * opt.hessianMultiplier 
                 if opt.newton then
                     stepSize = 1/torch.abs(eigenVal-eigenVal2)
                 end
                 parameters:add(eigenVec2 * stepSize)
                 --outputs_after = model:forward(inputs)
                 --cost_after = criterion:forward(outputs, targets)
                 cost_after = computeCurrentLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath) 
                 --print("cost_before")
                 --print(cost_before)
                 cost_before_acc[#cost_before_acc+1] = cost_before
                 --print("cost_after")
                 --print(cost_after)
                 cost_after_acc[#cost_after_acc+1] = cost_after
                 if cost_before > cost_after then flag = flag + 1 end
                 --sleep(2)
             end
             --print("eigenvalue")
             --print(eigenVal)
             --print("eigenvalue")
             --print(eigenVec)
             --torch.save("eigenVec_10-8.bin",eigenVec)
             ----os.exit()
         end
            powercallRecord[#powercallRecord+1] = flag
         end
         

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end

      norm_gradParam[#norm_gradParam + 1] = minibatch_norm_gradParam --accumulated every minibatch
      minibatch_norm_gradParam = 0  
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnist.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

----------------------------------------------------------------------
-- and train!
--
testErrTable = {} 
testAccTable = {}
timer = sys.clock()
while true do
   -- train/test  
   train(trainData)
   test(testData)

   torch.save("cost_before_acc.bin" , cost_before_acc);torch.save("cost_after_acc.bin",cost_after_acc)
   -- norm_gradParam's x-axis is the number of minibatches so far
   torch.save("norm_gradParam.bin", norm_gradParam)
   torch.save("eigenTable.bin",eigenTable)
   torch.save("eigenTableNeg.bin",eigenTableNeg)
   torch.save("powercallRecord.bin",powercallRecord)
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
   if epoch > opt.maxEpoch then 
       torch.save("time_it_took.bin",sys.clock()-timer)
       break 
   end 

end
