require 'nn'

local layer_size = 49

local model = nn.Sequential()
local image_size = 32

model:add(nn.Reshape(image_size*image_size))
model:add(nn.Linear(image_size*image_size, layer_size))
model:add(nn.Tanh())
model:add(nn.Linear(layer_size, image_size*image_size))
model:add(nn.Reshape(image_size, image_size))


return model
