require 'nn'
require 'optim'

local opt = [[
    -save  (default "logs")            subdirectory to save logs
    -m,--model  (default "cnn_mnist")  type of model for train: convnet|
    -c,--cuda (default false)          if true cuda enabled
    -l,--learningRate (default 0.05)   learning rate 
    -w,--weightDecay (default 0.0005) weight decay
    -b,--batchSize  (default 10)       batch size
    --saveInterval  (default 10)       save models per n epoch
    --max_epoch     (default 300)      maximum number of iterations
]]

--fix seed
torch.manualSeed(1234)


--load model 
if opt.model == 'cnn_mnist' then
    model = torch.load(dofile('models/'..opt.model..'.lua'))
end


--loss function
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--load dataset 
dataset = 

--obtain storage for parameters 
parameters, gradParameters = model:getParameters()

--set optimState
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}

--define train func
function train()
    model:training()
    epoch = epoch or 1
    
    -- keep tracking how much time it takes to learn per data sample
    local time = torch.tic()

    -- do one epoch
    print('<trainer> on training set')
    print('epoch #' .. epoch .. ' batchsize = ' .. opt.batchsize .. )
    
    --Get space for targets 
    local targets = torch.LongTensor(opt.batchSize)
    --creates random indices
    --:split(opt.batchSize) takes the first [opt.batchSize) number of indices
    --and puts them into a table. So, by :split(opt.batchSize), indices become
    --a table such that
    --{ 1: DoubleTensor - size: [opt.batchSize], 2: DoubleTensor - size: }
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
    -- remove the second element of the table (indices)
    indices[#indices] = nil

    local tic = torch.tic()
    for t,v in ipairs(indices) do
        -- Isn't indices already size 1?
        xlua.progress(t, #indices)
        
        local inputs = provider.trainData.data:index(1,v)
        targets:copy(provider.trainData.labels:index(1,v))

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            -- gradParameters will be updated / accumulcated
            model:backward(inputs, df_do)

            confusion:batchAdd(outputs, targets)

            return f,gradParameters
        end

        optim.sgd(feval, parameters, optimState)
    end

    confusion:updateValids() --calculates the accuracy things in confusion matrix

    print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1
end


function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print(c.blue '==>'.." testing")

    local bs = 125
    for i=1,provider.testData.data:size(1),bs do
        local outputs = model:forward(provider.testData.data:narrow(1,i,bs))
        confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs))
    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)

    -- save model every [saveInterval] epochs
    if epoch % opt.saveInterval == 0 then
        local filename = paths.concat(opt.save, 'model_' .. epoch)
        print('==> saving model to '..filename)
        torch.save(filename, model:clearState())
    end

    confusion:zero()
end



for i = 1, opt.max_epoch do
    train()
    test()
end


