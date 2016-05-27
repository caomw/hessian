require 'nn'
local nninit = require 'nninit'

nninit.sparse = function(module, tensor, sparsity)
  local nElements = tensor:nElement()
  local nSparseElements = math.floor(nElements - sparsity)
  local randIndices = torch.randperm(nElements):long()
  local sparseIndices = randIndices:narrow(1, 1, nSparseElements)

  -- Zero out selected indices
  tensor:view(nElements):indexFill(1, sparseIndices, 0)

  return module
end

local getBias = function(module)
  return module.bias
end


local model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 1000):init('weight', nninit.sparse, 15*1000):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 500):init('weight', nninit.sparse, 15*500):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(500, 250):init('weight', nninit.sparse, 15*250):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(250,30):init('weight', nninit.sparse, 15*30):init(getBias, nninit.constant, 0))
--model:add(nn.Sigmoid())
model:add(nn.Linear(30, 250):init('weight', nninit.sparse, 15*250):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(250, 500):init('weight', nninit.sparse, 15*500):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(500, 1000):init('weight', nninit.sparse, 15*1000):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Linear(1000, 28*28):init('weight', nninit.sparse, 15*784):init(getBias, nninit.constant, 0))
model:add(nn.Sigmoid())
model:add(nn.Reshape(28, 28))
------------------------------------------------------------

return model
