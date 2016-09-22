local d = require 'autograd'
local t = require 'torch'
d.optimize(true)

torch.manualSeed(1)

local input_size = 2
local hidden_size1 = 3
local output_size = 2

-- some data:
x = torch.randn(1,input_size)
y = torch.Tensor(1,output_size):zero() y[1][2] = 1

params = {
   W = {
      t.randn(input_size,hidden_size1),
      t.randn(hidden_size1,output_size)
   },
   b = {
      t.randn(hidden_size1),
      t.randn(output_size)
        }
}

-- define model
innerFn = function(params, x, y)
   local h1 = t.tanh(x * params.W[1] + params.b[1])
   local h2 = t.tanh(h1 * params.W[2] + params.b[2])
   local yHat = h2 - t.log(t.sum(t.exp(h2)))
   local loss = - t.sum(t.cmul(yHat, y))
   return loss
end

--local grads, loss = d(innerFn)(params, x,y)

v = {
   W = {
      t.randn(input_size,hidden_size1):fill(2),
      t.randn(hidden_size1,output_size):fill(3)
   },
   b = {
      t.randn(hidden_size1),
      t.randn(output_size)
        }
}

--grads = d(innerFn)(params, x, y)

--     local temp1 = 0
--     for i = 1, #grads.W do
--         temp1 = temp1 + t.sum(t.cmul(grads.W[i], v.W[i]))
--         temp2 = temp2 + t.sum(t.cmul(grads.b[i], v.b[i]))
--     end


-- local outerFn = function(params, x, y, v, grads)
--     --local product = clone(grads)
-- --     local flattened_grads = grads:view(grads:nElement()) 
-- --     local flattened_params = params:view(params:nElement())
-- --    return flattened_grads * flattened_params
-- --     local temp1 = 0
-- --     for i = 1, #grads.W do
-- --         temp1 = temp1 + t.sum(t.cmul(grads.W[i], v.W[i]))
-- --         temp2 = temp2 + t.sum(t.cmul(grads.b[i], v.b[i]))
-- --     end
--     local temp1 = t.sum(t.cmul(grads.W[1], v.W[1])) 
--     local temp2 = t.sum(t.cmul(grads.W[2], v.W[2]))
--     local temp3 = t.sum(t.cmul(grads.b[1], v.b[1])) 
--     local temp4 = t.sum(t.cmul(grads.b[2], v.b[2]))
--     local ans = temp1 + temp2 + temp3 + temp4
--     return ans
-- end

-- Hv, loss = d(outerFn)(params, x, y, v, grads)        

-- print(outerFn(params,x,y,v,grads))

--print(t.sum(t.cmul(grads.W[1].raw, v.W[1])) + t.sum(t.cmul(grads.W[2].raw, v.W[2])))


--grads(params, x, y)

--print(grads)
local ddf = d(function(params, x, y, v)
   local grads = d(innerFn)(params, x, y)
    
   local temp1 = t.sum(t.cmul(grads.W[1] , v.W[1])) + t.sum(t.cmul(grads.W[2] , v.W[2])) 
   local temp2 = t.sum(t.cmul(grads.b[1] , v.b[1])) + t.sum(t.cmul(grads.b[2] , v.b[2])) 
   return temp1 + temp2
end)

gradGrads = ddf(params, x, y, v) -- second order gradient of innerFn

print(gradGrads.W[1])


