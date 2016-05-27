require 'nn'

local model = nn.Sequential()

 ------------------------------------------------------------
-- convolutional network 
------------------------------------------------------------
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
-- model:add(nn.Dropout(opt.dropout_p))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- model:add(nn.Dropout(opt.dropout_p))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(64*2*2))
model:add(nn.Linear(64*2*2, 200))
model:add(nn.ReLU())
-- model:add(nn.Dropout(opt.dropout_p))
model:add(nn.Linear(200, 10))
------------------------------------------------------------

return model
