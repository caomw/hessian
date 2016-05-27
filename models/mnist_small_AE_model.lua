require 'nn'

local model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 30))
model:add(nn.Sigmoid())
--model:add(nn.Linear(1000, 250))
--model:add(nn.Sigmoid())
--model:add(nn.Linear(250,3))
--model:add(nn.Sigmoid())
--model:add(nn.Linear(3, 250))
--model:add(nn.Sigmoid())
--model:add(nn.Linear(250, 1000))
--model:add(nn.Sigmoid())
model:add(nn.Linear(30, 28*28))
model:add(nn.Sigmoid())
model:add(nn.Reshape(28, 28))
------------------------------------------------------------

return model
