local t = require 'torch'
torch.manualSeed(1)
local d = require 'autograd'
d.optimize(true)

local n = 5
--local diag = t.diag(torch.rand(n))
local A_temp = t.rand(n,n)*10
local A = A_temp:t() * A_temp
local b = t.rand(n, 1)*10
local c = t.rand(1)*10

local x = t.rand(n, 1)*10

-- define a quadratic function
function quadratic(x, A, b, c)
    local ans = 0.5 * t.transpose(x, 1,2) * A * x + t.transpose(b,1,2)*x + c
    return t.sum(ans)
end


local ddf = d(function(param, AA, bb, cc, vv)
    local grads = d(quadratic)(param, AA, bb, cc)
    local ans = t.sum(t.cmul(vv, grads))
    return ans
end)

local vvv = t.rand(n, 1)
local gradGrads = ddf(x, A, b, c, vvv)  

print(gradGrads)

print(A * vvv)


local function getHessian(x, A, b, c)
    -- define second-order diff function
    local getHessianVec = d(function(param, A, b, c, i)
        local grads = d(quadratic)(param, A, b, c)
        return t.sum(grads[i])
    end)

    local hessian_mat = torch.Tensor(n,n):zero()
    for i = 1, n do
        local hessian_vec = getHessianVec(x,A,b,c,i)
        local temp = hessian_mat:narrow(2, i, 1)
        temp:copy(hessian_vec)
    end
    return hessian_mat
end

print(getHessian(x, A, b, c))
print(A)


