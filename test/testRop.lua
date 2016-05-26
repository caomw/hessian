require 'nn'
require 'rop'

local input_size = 1000
local target_size = 1000
local hidden_size = 1000
local mini_batch_size = 10

local input = torch.randn(mini_batch_size, input_size)
local target = torch.ceil(torch.rand(mini_batch_size)*target_size)

local model = nn.Sequential()
   model:add(nn.Linear(input_size, hidden_size))
   model:add(nn.Sigmoid())
   model:add(nn.Linear(hidden_size, target_size))
   model:add(nn.LogSoftMax())
--local criterion = nn.MSECriterion()
local criterion = nn.ClassNLLCriterion()

-- We must collect the parameters, which also creates storage to store the
-- vector we will multiply with the Hessian, and to store the result (which the
-- R-op applied to the parameters).
local parameters, gradParameters, rParameters, rGradParameters = model:getParameters()
parameters:randn(parameters:size()) 


-- Set rParameters to the vector you want to multiply the Hessian with
rParameters:randn(parameters:size())

-- First do the normal forward and backward-propagation
local pred = model:forward(input)
local obj = criterion:forward(pred, target)

local df_do = criterion:backward(pred, target)
model:backward(input, df_do)

timer = torch.Timer() 
local model2 = model:clone()
local parameters2, gradParameters2 = model2:getParameters()


local d = rParameters:clone()
local epsilon = 10e-6
parameters2:copy(parameters + d * epsilon)
gradParameters2:zero()
local pred2 = model2:forward(input)
local obj2 = criterion:forward(pred,target)
local df_do2 = criterion:backward(pred,target)
model2:backward(input,df_do2)
local Hd = (gradParameters - gradParameters2)/epsilon
print('Time elapsed for finite difference: ' .. timer:time().real .. ' seconds')

timer = torch.Timer()
-- We calculate the R-ops as we go forward
local r_pred = model:rForward(input)

local rGradOutput =  criterion:rBackward(r_pred, target)
model:rBackward(input, torch.zeros(mini_batch_size, input_size), df_do, rGradOutput)

print('Time elapsed for rop: ' .. timer:time().real .. ' seconds')

-- The R-op applied to the parameters now contains the Hessian times the value
-- of rParameters
--print(rGradParameters)

--print(Hd)

print(torch.norm(torch.abs(rGradParameters) - torch.abs(Hd)))
print(Hd:size())
