require 'nn'

local model = nn.Sequential()
----------------------------------------------------------
----- single hidden layer MLP network 
------------------------------------------------------------
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
--model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
--model:add(nn.SpatialConvolutionMM(32, 32, 5, 5))
--model:add(nn.Tanh())
--model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(32*32))
model:add(nn.Linear(32*32, 500))
model:add(nn.Tanh())
model:add(nn.Linear(500, 500))
model:add(nn.Tanh())
model:add(nn.Linear(500, 10)) 
------------------------------------------------------------

return model
