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
--require 'cunn'

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
   --coefL2           (default 0)           L2 penalty on the weights
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
]]
torch.save("parameter_info.bin",opt)

local dataset_filepath =  opt.currentDir .. '/dataset-mnist32.lua' 
--print(dataset_filepath)
dofile(dataset_filepath)

local iterationMethods_filepath = opt.currentDir .. '/iterationMethods_classifier.lua'
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
   torch.setdefaulttensortype('torch.DoubleTensor')
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
cost_after_accH = {}
cost_after_accG = {}
eigenTable = {}
eigenTableNeg = {}
powercallRecord = {}
if opt.lineSearch  then
    lineSearchDecisionTable = {}
end
convergeTable1 = {}
convergeTable2 = {}

-- training function
function trainSGD(dataset)
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

         minibatch_norm_gradParam = torch.norm(gradParameters) 

         local clock = os.clock
         function sleep(n)  -- seconds
               local t0 = clock()
               while clock() - t0 <= n do end
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

      elseif opt.optimization == 'SGD'  then
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

      -- norm_gradParam[#norm_gradParam + 1] = minibatch_norm_gradParam --accumulated every minibatch
      -- minibatch_norm_gradParam = 0  
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
   local filename = paths.concat(opt.save, 'mnistClassifier_G.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)
   modelParam, _ = model:clone():getParameters()
   --torch.save(filename, model:clearState())
   torch.save(filename, modelParam)
   -- next epoch
   epoch = epoch + 1
end



-- training function
function trainHes(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSizeHessian .. ']')
   for t = 1,dataset:size(),opt.batchSizeHessian do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSizeHessian,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSizeHessian)
      local k = 1
      for i = t,math.min(t+opt.batchSizeHessian-1,dataset:size()) do
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

         minibatch_norm_gradParam = torch.norm(gradParameters) 

         local clock = os.clock
         function sleep(n)  -- seconds
               local t0 = clock()
               while clock() - t0 <= n do end
         end
	 local doGradStep = 1
   	 local v = -1
	 local stepSize = -1
 	 if torch.norm(gradParameters) < opt.gradNormThresh then
		-- First iteration method
		if opt.iterationMethod =="power" then
		    maxEigValH, v, converged1 = hessianPowermethodClassifier(inputs,targets,parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
		end
		if opt.iterationMethod =="lanczos" then
		    maxEigValH, v, converged1 = lanczosClassifier(inputs,targets,parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
		end
		convergeTable1[#convergeTable1+1] = converged1
		eigenTable[#eigenTable+1] = maxEigValH
		-- Second iteration method
		if opt.iterationMethod =="power" then  
		    minEigValH, v, converged2 = negativePowermethodClassifier(inputs,targets,parameters:clone(),opt.iterMethodDelta,opt.currentDir,maxEigValH,opt.modelpath)
		end
		if opt.iterationMethod =="lanczos"  then 
		    minEigValH, v, converged2 = negativeLanczosClassifier(inputs,targets,parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath,maxEigValH)
		end
		convergeTable2[#convergeTable2+1] = converged2
		eigenTableNeg[#eigenTableNeg+1] = minEigValH
		if minEigValH < 0 and converged1 and converged2 then --the Hessian has a reliable negative eigenvalue so we should proceed to this direction
		   doGradStep = 0;
		   cost_before = computeCurrentLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath) 
		   local searchTable = {2-6, 2^-4, 2^-2, 2^0, 2^2, 2^4, 2^6,
				       -2^-6, -2^-4, -2^-2, -2^0, -2^2, -2^4, -2^6}
		   local temp_loss = 10e8
		   for i=1,#searchTable do
			local linesearch_stepSize = searchTable[i]
			local loss_after = computeLineSearchLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath,v,linesearch_stepSize)
			if (loss_after - cost_before) < temp_loss then
			    id_record = i
			    temp_loss = loss_after - cost_before
			end
		   end
		   stepSize = searchTable[id_record]  
		   lineSearchDecisionTable[#lineSearchDecisionTable+1] = stepSize
		   parametersH = parameters:clone():add(v * stepSize) -- Hessian update
		   parametersG = parameters:clone():add(gradParameters * (-opt.learningRate)) -- gradient update
		   cost_afterH = computeCurrentLoss(inputs,targets,parametersH,opt.currentDir,opt.modelpath) 
		   cost_afterG = computeCurrentLoss(inputs,targets,parametersG,opt.currentDir,opt.modelpath) 
		   cost_before_acc[#cost_before_acc+1] = cost_before
		   cost_after_accH[#cost_after_accH+1] = cost_afterH
		   cost_after_accG[#cost_after_accG+1] = cost_afterG
	        end
	 end
 

         -- return f and df/dX
         return f,gradParameters, doGradStep, stepSize, v
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

      elseif opt.optimization == 'SGD'  then
	 -- Perform SGD step:
	 sgdState = sgdState or {
	    learningRate = opt.learningRate,
	    momentum = opt.momentum,
	    learningRateDecay = 5e-7
	 }
	 update(feval, parameters, sgdState)

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
   local filename = paths.concat(opt.save, 'mnistClassifier_H.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   -- torch.save(filename, model)
   modelParam, _ = model:clone():getParameters()
   --torch.save(filename, model:clearState())
   torch.save(filename, modelParam)
   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   bs = math.min(dataset:size(), opt.batchSize)
   for t = 1,dataset:size(),bs do
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
      for i = 1,bs do
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

flag = true
while flag do
   -- train/test  
   trainSGD(trainData)
   test(testData)
   if epoch > opt.maxEpoch then
	flag = false
   end
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end

flag = true
while flag do
   -- train/test  
   trainHes(trainData)
   test(testData)

   torch.save("cost_before_acc.bin" , cost_before_acc);
   torch.save("cost_after_accH.bin",cost_after_accH)
   torch.save("cost_after_accG.bin",cost_after_accG) 
   torch.save("norm_gradParam.bin", norm_gradParam)
   torch.save("eigenTable.bin",eigenTable)
   torch.save("eigenTableNeg.bin",eigenTableNeg)
   torch.save("powercallRecord.bin",powercallRecord)
   torch.save("convergeTable1.bin",convergeTable1)
   torch.save("convergeTable2.bin",convergeTable2)
   if opt.lineSearch  then
     torch.save("lineSearchDecision.bin",lineSearchDecisionTable)
   end
   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
   if epoch > opt.maxEpoch + opt.maxEpochHessian then
	flag = false
   end
end
torch.save("time_it_took.bin",sys.clock()-timer)
