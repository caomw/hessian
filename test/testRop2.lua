require 'nn'
local d = require 'autograd'
local t = require 'torch'

d.optimize(true)

torch.manualSeed(12)

local input_size = 2
local hidden_size1 = 3
local output_size = 2  

local mini_batch_size = 1

local x = torch.randn(1 ,input_size)
local y = torch.Tensor(1,output_size):zero() y[1][2] = 1  

local model = nn.Sequential() 
   model:add(nn.Linear(input_size, hidden_size1)) 
   model:add(nn.Sigmoid()) 
--   model:add(nn.Linear(hidden_size, hidden_size))  
--   model:add(nn.Sigmoid())                       
   model:add(nn.Linear(hidden_size1, hidden_size1)) 
   model:add(nn.Sigmoid())                         
   model:add(nn.Linear(hidden_size1, output_size))
   model:add(nn.LogSoftMax())                    
--local criterion = nn.MSECriterion()           
--local criterion = nn.ClassNLLCriterion() -- Negative Log-Likelihood  
local criterion = nn.CrossEntropyCriterion()


modelf, parameters = d.functionalize(model) 

params = { W = {}, b = {} }                 
params["W"][1] = parameters[1]:t():clone() 
params["b"][1] = parameters[2]:clone()    
params["W"][2] = parameters[3]:t():clone() 
params["b"][2] = parameters[4]:clone()    
params["W"][3] = parameters[5]:t():clone() 
params["b"][3] = parameters[6]:clone() 

-- define model                                                                                                                                                                    
local innerFn = function(params, input, target)            
   local h1 = t.sigmoid(input * params.W[1] + params.b[1]) 
   local h2 = t.sigmoid(h1 * params.W[2] + params.b[2])   
   --print(h2)                                           
   local h3 = h2 * params.W[3] + params.b[3]            
   local yHat = h3 - t.log(t.sum(t.exp(h3))) -- this is softmax 
   local loss = - t.sum(t.cmul(yHat, target))
   return loss
   -- http://cs231n.github.io/linear-classify/#loss   
end 

print("autograd loss") 
print(innerFn(params, x, y)) 

--os.exit()

-- Creating the v vector as in Hv
v = { W = {}, b = {}}
v["W"][1] = torch.randn(parameters[1]:t():size())*10
v["b"][1] = torch.randn(parameters[2]:size())*10
v["W"][2] = torch.randn(parameters[3]:t():size())*10 
v["b"][2] = torch.randn(parameters[4]:size())*10   
v["W"][3] = torch.randn(parameters[5]:t():size())*10 
v["b"][3] = torch.randn(parameters[6]:size())*10 


local ddf = d(function(params, x, y, v)       
   local grads = d(innerFn)(params, x, y)    
   local temp1 = t.sum(t.cmul(grads.W[1] , v.W[1])) + t.sum(t.cmul(grads.W[2] , v.W[2]))  
   local temp2 = t.sum(t.cmul(grads.b[1] , v.b[1])) + t.sum(t.cmul(grads.b[2] , v.b[2])) 
   return temp1 + temp2                         
end) 
    
Hv_autograd = ddf(params, x, y, v) -- second order gradient of innerFn   

print(Hv_autograd.W[1])

print(model)

local parameters, gradParameters = model:getParameters()

local pred = model:forward(x) 
maxs, indices = torch.max(y,2)
local obj = criterion:forward(pred, indices)
print(obj)
local df_do = criterion:backward(pred, indices)
model:backward(x, df_do)

local model2 = model:clone() 
local parameters2, gradParameters2 = model2:getParameters()  
print(v)

local temp = {}
for i = 1, #v.W do
    temp[i] = torch.cat(v["W"][i]:view(v["W"][i]:nElement()),v["b"][i])
end
--print(temp)
local v_concat = torch.cat(torch.cat(temp[1],temp[2]), temp[3])
--local d_vec = 
local  epsilon = 10e-6
parameters2:copy(parameters + v_concat * epsilon)
gradParameters2:zero()
local pred2 = model2:forward(x)
maxs, indices = torch.max(y,2)
local obj2 = criterion:forward(pred,indices)
local df_do2 = criterion:backward(pred,indices)
model2:backward(x,df_do2)
local Hv_finite = (gradParameters - gradParameters2)/epsilon  
print("Hv_finite")
print(Hv_finite)
print("Hv_autograd")
local temp = {}
for i = 1, #Hv_autograd.W do
    temp[i] = torch.cat(Hv_autograd["W"][i]:view(Hv_autograd["W"][i]:nElement()),Hv_autograd["b"][i])
end
local Hv_aut_con = torch.cat(torch.cat(temp[1],temp[2]), temp[3])
print(Hv_aut_con)


--TODO hard-corded the model architecture so should fix it when I have time.
local function getHessian(params, input, target)
    local getHessianVec = d(function(params, input, target, i)
        local grads = d(innerFn)(params, input, target)
        --local temp = {}
        --for j = 1, #grads.W do
        --    temp[j] = torch.cat(grads["W"][j]:view(grads["W"][j]:nElement()),grads["b"][j])
        --end
        --local grads_concat = torch.cat(torch.cat(temp[1],temp[2]), temp[3]) 
        local grads_concat = flattenParams(grads)
        print("printing grads_concat")
        print(grads_concat)
        return grads_concat[i]
    end)

    local n = 29
    local hessian_mat = torch.Tensor(n,n):zero()
    for i = 1, n do
        print(params) print(input) print(target) print(i)
        local hes_vec = getHessianVec(params,input,target,i)
        local temp_hes_mat = hessian_mat:narrow(2, i, 1)
        local temp = {}
        local hes_vec_concat
        for j = 1, #hes_vec.W do
            temp[j] = torch.cat(hes_vec["W"][j]:view(hes_vec["W"][j]:nElement()),hes_vec["b"][j])
            if j == 1 then
               hes_vec_concat = temp[j]
           else
               hes_vec_concat = torch.cat(hes_vec_concat, temp[j])
           end
        end
        --local hes_vec_concat = torch.cat(torch.cat(temp[1],temp[2]), temp[3]) 
        --print("printing hes_vec_concat")
        --print(hes_vec_concat)
        temp_hes_mat:copy(hes_vec_concat)
    end
    return hessian_mat
end

-- input:  params table s.t. params = {W={},b={}} 
-- output: param_flattend ; type: Tensor
function flattenParams(params)
    local temp = {}
    local total_length = 0
    local param_flattened
    for i = 1, #params.W do
        temp[i] = torch.cat(params["W"][i]:view(params["W"][i]:nElement()),params["b"][i])
        total_length = total_length + temp[i]:nElement()
        if i == 1 then
            param_flattened = temp[i]
        else
            param_flattened = torch.cat(param_flattened, temp[i])
        end
    end
    --local param_flattened = torch.zero(torch.Tensor(total_length))
    --for i = 1, #params["W"] do
    --    local ith_param_size = params["W"][i]:nElement() + params["b"][i]:nElement()
    --    local temp_param = param_flattened:narrow(1, (i-1)*ith_param_size+1, ith_param_size)
    --    print(temp[i]:size()) print(temp_param:size())
    --    temp_param:copy(temp[i])
    --end
    return param_flattened
end

--print("testing out flattenParams function")
--print(flattenParams(Hv_autograd))
--os.exit()
        

--print(getHessian(params,x,y))
local Hessian = getHessian(params,x,y)
print(Hessian)
--os.exit()
--print(v_concat:double():view(29,1))

print("printing Hessian * v")
print(Hessian * v_concat:double():view(29,1))
print("printing Hv_autograd")
print(Hv_aut_con)
print("printing Hv_finite")
print(Hv_finite)
--print(Hessian)

local e, V = torch.symeig(Hessian)

local lam_max = torch.max(e)
local lam_min = torch.min(e)
--print("printing condition number")
--print(lam_max) print(lam_min)
--print(lam_max / lam_min)
