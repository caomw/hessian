require 'nn'

local model = nn.Sequential()
------------------------------------------------------------
-- convolutional network
------------------------------------------------------------
-- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMap(nn.tables.random(3,16,1), 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 2 : filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMap(nn.tables.random(16, 256, 4), 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.Dropout(0.5))
-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(256*5*5))
--model:add(nn.Linear(256*5*5, 128))
--model:add(nn.ReLU())
--model:add(nn.Dropout(0.5))
model:add(nn.Linear(256*5*5,10))

return model
