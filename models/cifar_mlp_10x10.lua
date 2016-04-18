require 'nn'

local model = nn.Sequential()
----------------------------------------------------------
----- multi-layer perceptron network 
------------------------------------------------------------
model:add(nn.Reshape(3*10*10))
model:add(nn.Linear(3*10*10, 50))
model:add(nn.Tanh())
model:add(nn.Linear(50, 10))


return model

