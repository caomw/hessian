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
   --mu_max (default 0.9)                   for momentum scheduling
]]
torch.save("parameter_info.bin",opt)

local dataset_filepath =  opt.currentDir .. '/dataset-mnist.lua' 
--print(dataset_filepath)
dofile(dataset_filepath)

--local iterationMethods_filepath = opt.currentDir .. '/iterationMethods_AE_cuda.lua'
--dofile(iterationMethods_filepath)


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
----------------------------------------------------------------------
-- loss function: MSE

criterion = nn.MSECriterion():cuda()
criterion.sizeAverage = false

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
--trainData.data:mul(255)
--trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = mnist.loadTestSet(nbTestingPatches, geometry)
--testData.data:mul(255)
--testData:normalizeGlobal(mean, std)

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

   currentLoss = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.CudaTensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.CudaTensor(opt.batchSize)
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
         local f = criterion:forward(outputs, inputs)
         --print("cost in general")
         

         -- estimate df/dW
         local df_do = criterion:backward(outputs, inputs)
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

	 if opt.clipGrad == true then
		if torch.norm(gradParameters) > opt.max_grad_norm then
    			local shrink_factor = opt.max_grad_norm / torch.norm(gradParameters)
    			gradParameters:mul(shrink_factor)
		end
	end
 
        f = f/inputs:size(1)
        gradParameters:div(inputs:size(1))

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
	         learningRateDecay = 2e-6
	     }
         if epoch == 3000 then 
             sgdState.learningRate = sgdState.learningRate/10 
             local filename = paths.concat(opt.save, 'mnistAE_epoch3000.net')
             local modelParam, _ = model:clone():getParameters()
             torch.save(filename, modelParam)
         end
         sgdState.momentum =  math.min(1 - math.pow(2,-1-math.log(torch.floor(epoch / 250) + 1)/math.log(2)), opt.mu_max)
	     _, fs = optim.sgd(feval, parameters, sgdState)
         currentLoss = currentLoss + fs[1]
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
   --print(confusion)

   local numBatches = dataset:size()/opt.batchSize
   currentLoss = currentLoss / numBatches
   trainLogger:add{['MSE (train set)'] = currentLoss}
   print("<trainer> MSE (train set) = " .. currentLoss)
   --confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnistAE_G.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
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
   ccost = 0
   -- local vars
   local time = sys.clock()

   currentLoss = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSizeHessian .. ']')
   for t = 1,dataset:size(),opt.batchSizeHessian do
      -- create mini batch
      local inputs = torch.CudaTensor(opt.batchSizeHessian,1,geometry[1],geometry[2])
      local targets = torch.CudaTensor(opt.batchSizeHessian)
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
         local f = criterion:forward(outputs, inputs)
         --print("cost in general")
         

         -- estimate df/dW
         local df_do = criterion:backward(outputs, inputs)
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

         f = f/inputs:size(1)
         gradParameters:div(inputs:size(1))

         minibatch_norm_gradParam = torch.norm(gradParameters) 

         local clock = os.clock
         function sleep(n)  -- seconds
               local t0 = clock()
               while clock() - t0 <= n do end
         end
	 local doGradStep = 1
   	 local v = -1
	 local stepSize = -1
         --print("gradNorm = " ..torch.norm(gradParameters))
 	 if torch.norm(gradParameters) < opt.gradNormThresh then
	     --flag = flag + 1
	     -- First iteration method
	     if opt.iterationMethod =="power" then
		 maxEigValH, v, converged1 = hessianPowermethodAE(inputs, parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
	     end
	     if opt.iterationMethod =="lanczos" then
		 maxEigValH, v, converged1 = lanczosAE(inputs, parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
	     end
	     convergeTable1[#convergeTable1+1] = converged1
	     eigenTable[#eigenTable+1] = maxEigValH
             collectgarbage()
	     -- Second iteration method
	     if opt.iterationMethod =="power" then  
		 minEigValH, v, converged2 = negativePowermethodAE(inputs, parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath,maxEigValH)
	     end
	     if opt.iterationMethod =="lanczos"  then
		 minEigValH, v, converged2 = negativeLanczosAE(inputs, parameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath,maxEigValH)
	     end
	     convergeTable2[#convergeTable2+1] = converged2
             --print(converged1)
             --print(converged2)
	     eigenTableNeg[#eigenTableNeg+1] = minEigValH
	     if minEigValH < 0 and converged1 and converged2 then --the Hessian has a reliable negative eigenvalue so we should proceed to this direction
		 local doGradStep = 0;
		 --flag = flag + 1
                 local gradStepSize = torch.norm(gradParameters) * opt.learningRate
                 --print("opt.learningRate = " .. opt.learningRate)
                 --print("gradStepSize = " .. gradStepSize)
		 local cost_before = computeCurrentLossAE(inputs,inputs,parameters:clone(),opt.currentDir,opt.modelpath) 
		 if opt.lineSearch then
		     local searchTable = {2^-7, 2^-6, 2^-5, 2^-4, 2^-3, 2^-2, 2^-1, 2^0, 2^1, 2^2, 2^3, gradStepSize,
		                          -2^-7, -2^-6, -2^-5, -2^-4, -2^-3, -2^-2, -2^-1, -2^0, -2^1, -2^2, -2^3, -gradStepSize}
		     local temp_loss = 10e8
                     --print("cost_before = " .. cost_before)
                     --print("gradStepSize = " .. gradStepSize)
		     for i=1,#searchTable do
		        local linesearch_stepSize = searchTable[i]
		        local loss_after = computeLineSearchLossAE(inputs,inputs,parameters:clone(),opt.currentDir,opt.modelpath,v,linesearch_stepSize)
                        --print("linesearch_stepSize  = " .. linesearch_stepSize)
                        --print("loss_after = " .. loss_after)
		        if (loss_after - cost_before) < temp_loss then
		            id_record = i
		            temp_loss = loss_after - cost_before
		        end
		    end
		    stepSize = searchTable[id_record]  
		    lineSearchDecisionTable[#lineSearchDecisionTable+1] = stepSize
		 end
		 local parametersH = parameters:clone():add(v * stepSize) -- Hessian update
		 local parametersG = parameters:clone():add(gradParameters * (-opt.learningRate)) -- gradient update
		 local cost_afterH = computeCurrentLossAE(inputs,inputs,parametersH,opt.currentDir,opt.modelpath) 
		 local cost_afterG = computeCurrentLossAE(inputs,inputs,parametersG,opt.currentDir,opt.modelpath) 
                 parametersG = nil
                 parametersH = nil
		 cost_before_acc[#cost_before_acc+1] = cost_before
		 cost_after_accH[#cost_after_accH+1] = cost_afterH
		 cost_after_accG[#cost_after_accG+1] = cost_afterG
                 print("cost_before = " .. cost_before)
                 print("cost_afterH = " .. cost_afterH)
                 print("cost_afterG = " .. cost_afterG)
		 --if cost_before > cost_after then flag = flag + 1 end
		 --sleep(2)
	     end
	 end
         --powercallRecord[#powercallRecord+1] = flag
         --end
         

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
	    learningRateDecay = 5e-6
	 }
	 _, fs = update(feval, parameters, sgdState)
         --print("fs = " .. fs[1])
         currentLoss = currentLoss + fs[1]
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
   --print(confusion)
   local numBatches = dataset:size()/opt.batchSizeHessian
   --print("numBatches = " .. numBatches)
   currentLoss = currentLoss / numBatches
   trainLogger:add{['MSE (train set)'] = currentLoss}
   print("<trainer> MSE (train set) = " .. currentLoss)
   --confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'mnistAE_H.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
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
   currentLoss = 0
   local bs = math.min(dataset:size(), opt.batchSize)
   for t = 1,dataset:size(),bs do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.CudaTensor(bs,1,geometry[1],geometry[2])
      local targets = torch.CudaTensor(bs)
      local k = 1
      for i = t,math.min(t+bs-1,dataset:size()) do
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
      testLoss = criterion:forward(preds, inputs)
      testLoss = testLoss/inputs:size(1)

      currentLoss = currentLoss + testLoss
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   --print(confusion)
   local numBatches = dataset:size()/bs
   currentLoss = currentLoss /  numBatches
   print("<trainer> MSE (test set) = " .. currentLoss)
   testLogger:add{['% MSE (test set)'] = currentLoss}
   --confusion:zero()
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
      trainLogger:style{['% MSE (train set)'] = '-'}
      testLogger:style{['% MSE (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end
print('***********************************************:')
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
      trainLogger:style{['% MSE (train set)'] = '-'}
      testLogger:style{['% MSE (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
   if epoch > opt.maxEpoch + opt.maxEpochHessian then
	flag = false
   end
end
torch.save("time_it_took.bin",sys.clock()-timer)
