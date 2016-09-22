local d = require 'autograd'
local t = require 'torch'
require 'nn'

torch.manualSeed(1)

d.optimize(true)

local input_size = 2
local hidden_size1 = 3
local output_size = 2

local mini_batch_size = 1

local input = torch.randn(1,input_size)
--print("input:") print(input)
local target = torch.Tensor(1,output_size):fill(0) 
target[1][2] = 1

local model = nn.Sequential()      
   model:add(nn.Linear(input_size, hidden_size1))  
   model:add(nn.Sigmoid())                       
--   model:add(nn.Linear(hidden_size, hidden_size))   
--   model:add(nn.Sigmoid())                 
   model:add(nn.Linear(hidden_size1, hidden_size1))  
   model:add(nn.Sigmoid())                      
   model:add(nn.Linear(hidden_size1, output_size))  
   model:add(nn.LogSoftMax())               
--local criterion = nn.MSECriterion()       
--local criterion = nn.ClassNLLCriterion() -- Negative Log-Likelihood 
local criterion = nn.CrossEntropyCriterion()

--print(model)

modelf, parameters = d.functionalize(model)

--print(parameters)

params = { W = {}, b = {} }
params["W"][1] = parameters[1]:t():clone()
params["b"][1] = parameters[2]:clone()
params["W"][2] = parameters[3]:t():clone()
params["b"][2] = parameters[4]:clone()
params["W"][3] = parameters[5]:t():clone()
params["b"][3] = parameters[6]:clone()

--print(params)

-- define model
local innerFn = function(params, input, target)
   local h1 = t.sigmoid(input * params.W[1] + params.b[1])
   local h2 = t.sigmoid(h1 * params.W[2] + params.b[2])
   --print(h2)
   local h3 = h2 * params.W[3] + params.b[3]
   local yHat = h3 - t.log(t.sum(t.exp(h3))) -- this is softmax
   local loss = - t.sum(t.cmul(yHat, target)) -- and its loss -- this equation suggests that target has to be one-hot vector / otherwise produce incorrect values
   --local loss = d.loss.crossEntropy(yHat, target)
   local loss = d.loss.crossEntropy(h3, target)
   return loss 
   -- http://cs231n.github.io/linear-classify/#loss
end


print("autograd loss")
print(innerFn(params, input, target))


local computeLoss = function(input, target)
    local f = model:forward(input)
    local loss = criterion:forward(f:view(f:nElement()),target)
    return loss
end

--target:add(1)

print("torch loss")
maxs, indices = torch.max(target,2) --if 1x10 vector, dim=2 takes the max among 10 classes
print(computeLoss(input,indices))
--print("torch nn modeul after LogSoftMax")
--print(model:get(6).output)
-- TODO: Why does innerFn and computeLoss produce two different loss values... 
-- h3 and f is exactly the same values. 
-- TODO: Look at the code of nn.LogSoftMax 
-- Both TODO done.
