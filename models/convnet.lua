require 'nn'

local cnn = nn.Sequential()

-- conv1
cnn:add(nn.SpatialConvolution(1, 10, 5, 5, 1, 1, 0, 0))   --28x28
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))                             -- 14x14
cnn:add(nn.Dropout(0.2))
-- conv2
cnn:add(nn.SpatialConvolution(10, 20, 5, 5, 1, 1, 0, 0))  -- 10x10
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2))                             --  5x5

-- fully connected
cnn:add(nn.SpatialConvolution(20, 120, 5, 5, 1, 1, 0, 0)) --  1x1
cnn:add(nn.ReLU())
cnn:add(nn.Reshape(120)) -- probably 120x1x1 so this makes it just 120 vector?
cnn:add(nn.Linear(120, 84))
cnn:add(nn.Dropout(0.5)) ---dropout added on 9/4
cnn:add(nn.Linear(84, 10))

---- Loss Function ----

cnn:add(nn.LogSoftMax()) -- negative log-liklihood

return cnn
