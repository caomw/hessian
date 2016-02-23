require 'nn'
require 'optim'

local opt = [[
    -m,--model  (default "cnn_mnist")  type of model for train: convnet|
    -c,--cuda (default false)          if true cuda enabled
    -l,--learningRate (default 0.05)   learning rate 
    -b,--batchSize  (default 10)       batch size
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



--set optimState
optimState = {
    learningRate = opt.learningRate,
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
    for t = 1, 


    
