require 'nn'

local args = lapp[[
  --modelpath (string)
]]

net=nn.Sequential()
net:add(dofile(args.modelpath))
param,grad=net:getParameters()

print("parameter size of this model:")
print(param:size())



