require 'nn'
function hessianPowermethod(inputs,targets,param,gradParam, delta, filepath, modelpath) 
    local epsilon = 10e-8 -- for numerical differentiation to get Hd 
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    local norm_Hd = 1; local norm_Hd_old = 2
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    while (norm_Hd_old - norm_Hd) > delta do
        --print("hessianPowermethod")
        --print((norm_Hd_old-norm_Hd))
        print("param_new size ") ; print(param_new:size())
        print("param size "); print(param:size())
        print("vector d size"); print(d:size())
        param_new:copy(param + d * epsilon)

        --reset gradients
        gradParam_eps:zero()

        --feedforward and backpropagation
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do) --gradParams_eps should be updated here 

        Hd = (gradParam_eps - gradParam) / epsilon
        norm_Hd_old = norm_Hd
        norm_Hd = torch.norm(Hd)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d = Hd / norm_Hd
        epsilon = 2*torch.sqrt(10e-7)*(1 + torch.norm(param))/torch.norm(d)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
    end

    --print("sanity check")
    --print(torch.cdiv(Hd,d))
    return d , torch.cdiv(Hd,d)[1] 
end

function negativePowermethod(inputs,targets,param,gradParam, delta, filepath,eigen,modelpath) 
    local epsilon = 10e-8 -- for numerical differentiation to get Hd
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    local norm_Md = 1; local norm_Md_old = 2 
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    while (norm_Md_old-norm_Md) > delta do
        --print("negativePowermethod")
        --print((norm_Md_old - norm_Md))
        param_new:copy(param + d * epsilon)

        --reset gradients
        gradParam_eps:zero()

        --feedforward and backpropagation
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do) --gradParams_eps should be updated here 

        local Hd = (gradParam_eps - gradParam) / epsilon
        Md = d * eigenVal - Hd
        norm_Md_old = norm_Md
        norm_Md = torch.norm(Md)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d = Md / norm_Md
        epsilon = 2*torch.sqrt(10e-7)*(1 + torch.norm(param))/torch.norm(d)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
    end 

    --print("sanity check")
    --print(torch.cdiv(Md,d))
    return d , torch.cdiv(Md,d)[1] 
end

function computeCurrentLoss(inputs,targets,parameters,filepath,modelpath)
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()

    local param_new,gradParam_eps = model:getParameters() --I need to do this
    param_new:copy(parameters)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)
    df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do) --gradParams_eps should be updated here 

    return loss
end

function computeLineSearchLoss(inputs,targets,parameters,filepath,modelpath,eigenVector,stepSize)
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()

    local param_new,gradParam_eps = model:getParameters() --I need to do this
    param_new:copy(parameters)
    param_new:add(eigenVector*stepSize)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)
    df_do = criterion:backward(outputs, targets)
    model:backward(inputs, df_do) --gradParams_eps should be updated here 

    return loss
end


function checkModelID()
    model = nn.Sequential()
    model:add(dofile('models/train-mnist-model.lua'))
    id = torch.pointer(model)

    
    return id
end

