require 'nn'

local model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 500))
model:add(nn.Sigmoid())
model:add(nn.Linear(500, 250))
model:add(nn.Sigmoid())
model:add(nn.Linear(250,30))
--model:add(nn.Sigmoid())
model:add(nn.Linear(30, 250))
model:add(nn.Sigmoid())
model:add(nn.Linear(250, 500))
model:add(nn.Sigmoid())
model:add(nn.Linear(500, 1000))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 28*28))
model:add(nn.Sigmoid())
model:add(nn.Reshape(28, 28))
------------------------------------------------------------

return model
