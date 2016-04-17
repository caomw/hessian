----------------------------------------------------------------------
-- This script shows how to train different models on the CIFAR
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
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

require 'nn'
require 'optim'
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
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
cmd:option('-gradnormThresh', 0.5, 'threshold of grad norm to switch from gradient descent to hessian')
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

local iterationMethods_filepath = opt.currentDir .. '/iterationMethods.lua'
dofile(iterationMethods_filepath)

local update_filepath = opt.currentDir .. '/update.lua'
dofile(update_filepath)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()
   model:add(dofile(opt.currentDir .. opt.modelpath))
else
   print('<trainer> reloading previously trained network')
   model = nn.Sequential()
   model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<cifar> using model:')
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
   trsize = 50000
   tesize = 10000
else
   trsize = 2000
   tesize = 1000
end

-- download dataset
--if not paths.dirp('cifar-10-batches-t7') then
--   local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
--   local tar = paths.basename(www)
--   os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
--end

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
local dataset_filepath = opt.currentDir .. '/../data/cifar-10-batches-t7'
for i = 0,4 do
   subset = torch.load(dataset_filepath .. '/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load(dataset_filepath .. '/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

if opt.preprocess then
  ----------------------------------------------------------------------
  -- preprocess/normalize train/test sets
  --
  
  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()
  
  -- preprocess trainSet
  normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  mean_u = trainData.data[{ {},2,{},{} }]:mean()
  std_u = trainData.data[{ {},2,{},{} }]:std()
  trainData.data[{ {},2,{},{} }]:add(-mean_u)
  trainData.data[{ {},2,{},{} }]:div(-std_u)
  -- normalize v globally:
  mean_v = trainData.data[{ {},3,{},{} }]:mean()
  std_v = trainData.data[{ {},3,{},{} }]:std()
  trainData.data[{ {},3,{},{} }]:add(-mean_v)
  trainData.data[{ {},3,{},{} }]:div(-std_v)
  
  -- preprocess testSet
  for i = 1,testData:size() do
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
  end
  -- normalize u globally:
  testData.data[{ {},2,{},{} }]:add(-mean_u)
  testData.data[{ {},2,{},{} }]:div(-std_u)
  -- normalize v globally:
  testData.data[{ {},3,{},{} }]:add(-mean_v)
  testData.data[{ {},3,{},{} }]:div(-std_v)
end

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))


-- display function
function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter % 10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{
            image=model:get(1).weight, zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='stage 1: weights', padding=1
         }
         win_w2 = image.display{
            image=model:get(4).weight, zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='stage 2: weights', padding=1
         }
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{
            image=W1, zoom=0.5, min=-1, max=1,
            win=win_w1, legend='W1 weights'
         }
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{
            image=W2, zoom=0.5, min=-1, max=1,
            win=win_w2, legend='W2 weights'
         }
      end
   end
   iter = iter + 1
end

norm_gradParam = {}
minibatch_norm_gradParam = 0
cost_before_acc = {}
cost_after_accH = {}
cost_after_accG = {}
eigenTable = {}
eigenTableNeg = {}
powercallRecord = {}
if opt.lineSearch then
    lineSearchDecisionTable = {}
end
convergeTable1 = {} 
convergeTable2 = {} 

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local input = dataset.data[i]
         local target = dataset.labels[i]
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,#inputs do
            -- estimate f
            local output = model:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            model:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])

            -- visualize?
            if opt.visualize then
               display(inputs[i])
            end
         end

         -- normalize gradients and f(X)
         gradParameters:div(#inputs)
         f = f/#inputs
         trainError = trainError + f

         --record the norm of gradients
         minibatch_norm_gradParam = torch.norm(gradParameters)

         local clock = os.clock
         function sleep(n) --seconds
             local t0 = clock()
             while clock() - t0 <= n do end
         end

         -- hessian mode?
         local doGradStep = 1
         local v = -1
         local stepSize = -1
         if opt.hessian then
            local flag = 0
            if torch.norm(gradParameters) < opt.gradnormThresh then
             flag = flag + 1
             -- First iteration method
             if opt.iterationMethod =="power" then
                 maxEigValH, v, converged1 = hessianPowermethod(inputs,targets,parameters:clone(),gradParameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
             end
             if opt.iterationMethod =="lanczos" then
                 maxEigValH, v, converged1 = lanczos(inputs,targets,parameters:clone(),gradParameters:clone(),opt.iterMethodDelta,opt.currentDir,opt.modelpath)
             end
             convergeTable1[#convergeTable1+1] = converged1
             eigenTable[#eigenTable+1] = maxEigValH
             -- Second iteration method
             if opt.iterationMethod =="power" then
                 minEigValH, v, converged2 = negativePowermethod(inputs,targets,parameters:clone(),gradParameters:clone(),opt.iterMethodDelta,opt.currentDir,maxEigValH,opt.modelpath)
             end
             if opt.iterationMethod =="lanczos"  then
                 minEigValH, v, converged2 = negativeLanczos(inputs,targets,parameters:clone(),gradParameters:clone(),opt.iterMethodDelta,opt.currentDir,maxEigValH,opt.modelpath)
             end
             convergeTable2[#convergeTable2+1] = converged2
             eigenTableNeg[#eigenTableNeg+1] = minEigValH
             if minEigValH < 0 and converged1 and converged2 then --the Hessian has a reliable negative eigenvalue so we should proceed to this direction
                doGradStep = 0;
                flag = flag + 1
                cost_before = computeCurrentLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath)
                stepSize = opt.learningRate * opt.hessianMultiplier
                if opt.newton then
                    stepSize = 1/torch.abs(minEigValH)
                end     
                if opt.lineSearch then
                    local searchTable = {2^0, 2^1, 2^2, 2^3, 2^4, 2^5
                                      -2^0, -2^1, -2^2, -2^3, -2^4, -2^5}
                    local temp_loss = 10e8   
                    for i=1,#searchTable do
                        local linesearch_stepSize = opt.learningRate * searchTable[i]
                        local loss_after = computeLineSearchLoss(inputs,targets,parameters:clone(),opt.currentDir,opt.modelpath,v,linesearch_stepSize)
                        if (loss_after - cost_before) < temp_loss then
                            id_record = i
                            temp_loss = loss_after - cost_before
                        end     
                    end     
                    stepSize = opt.learningRate * searchTable[id_record]
                    lineSearchDecisionTable[#lineSearchDecisionTable+1] = stepSize
                end    
                parametersH = parameters:clone():add(v * stepSize) -- Hessian update
                parametersG = parameters:clone():add(gradParameters * (-opt.learningRate)) -- gradient update
                cost_afterH = computeCurrentLoss(inputs,targets,parametersH,opt.currentDir,opt.modelpath) 
                cost_afterG = computeCurrentLoss(inputs,targets,parametersG,opt.currentDir,opt.modelpath) 
                cost_before_acc[#cost_before_acc+1] = cost_before
                cost_after_accH[#cost_after_accH+1] = cost_afterH
                cost_after_accG[#cost_after_accG+1] = cost_afterG
             end -- of "if minEigValH < 0 and converged1 and converged2 then"
            end -- of "if torch.norm(gradParameters) < opt.gradnormThresh then"
            powercallRecord[#powercallRecord+1] = flag
         end --of "if opt.hessian then"
          
         -- return f and df/dX
         return f,gradParameters, doGradStep, stepSize, v
    end
 

    -- optimize on current mini-batch
    if opt.optimization == 'CG' then
       config = config or {maxIter = opt.maxIter}
       optim.cg(feval, parameters, config)
    elseif opt.optimization == 'LBFGS' then
       config = config or {learningRate = opt.learningRate,
                           maxIter = opt.maxIter,
                           nCorrection = 10}
       optim.lbfgs(feval, parameters, config)
    elseif opt.optimization == 'SGD' then
       config = config or {learningRate = opt.learningRate,                         
                           weightDecay = opt.weightDecay,
                           momentum = opt.momentum,
                           learningRateDecay = 5e-7}
       --optim.sgd(feval, parameters, config)
       update(feval, parameters, config)
    elseif opt.optimization == 'ASGD' then
       config = config or {eta0 = opt.learningRate,
                           t0 = nbTrainingPatches * opt.t0}
       _,_,average = optim.asgd(feval, parameters, config)
    else
       error('unknown optimization method')
    end

    norm_gradParam[#norm_gradParam + 1] = minibatch_norm_gradParam  --accumulated every minibatch
    minibatch_norm_gradParam = 0
 end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   local trainAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'cifar.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   --torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)
   -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size() do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- get new sample
      local input = dataset.data[t]
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
      confusion:add(pred, target)

      -- compute error
      err = criterion:forward(pred, target)
      testError = testError + err
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / dataset:size()

   -- print confusion matrix
   print(confusion)
   local testAccuracy = confusion.totalValid * 100
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end

----------------------------------------------------------------------
-- and train!
--
testErrTable = {}
testAccTable = {}
trainErrTable = {}
trainAccTable = {}
timer = sys.clock()      
while true do
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)

   torch.save("cost_before_acc.bin" , cost_before_acc)
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
   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}
   
   testErrTable[#testErrTable+1] = testErr; testAccTable[#testAccTable+1] = testAcc
   trainErrTable[#trainErrTable+1] = trainErr; trainAccTable[#trainAccTable+1] = trainAcc

   -- plot logger
   if opt.plot then
     accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
     errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
     accLogger:plot()
     errLogger:plot()
   end
   if epoch > opt.maxEpoch then 
       torch.save("testErr.bin",testErrTable);torch.save("testAcc.bin",testAccTable)
       torch.save("trainErr.bin",trainErrTable);torch.save("trainAcc.bin",trainAccTable)
       torch.save("time_it_took.bin",sys.clock()-timer)
       break 
   end
end
