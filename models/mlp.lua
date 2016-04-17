require 'nn'

local model = nn.Sequential()
----------------------------------------------------------
----- multi-layer perceptron network 
------------------------------------------------------------
model:add(nn.Reshape(32*32))
model:add(nn.Linear(32*32, 50))
model:add(nn.Tanh())
model:add(nn.Linear(50, 10))


return model

