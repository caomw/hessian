require 'nn'
function hessianPowermethod(inputs,targets,param,gradParam, delta, filepath, modelpath) 
    local max_iter = 30
    local epsilon = 10e-8 -- for numerical differentiation to get Hd 
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    --local norm_Hd = 1; local norm_Hd_old = 2
    local min_val = 1; local max_val = 2
    local sanity_vector = 0
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    --while (norm_Hd_old - norm_Hd) > delta do
    local i = 0
    while (max_val - min_val) > delta do
        --print("hessianPowermethod")
        --print((norm_Hd_old-norm_Hd))
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
        sanity_vector = torch.cdiv(Hd,d)
        max_val = torch.max(sanity_vector) 
        min_val = torch.min(sanity_vector)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
        i = i + 1
        if i > max_iter then break end
    end

    print("sanity check")
    --print(torch.cdiv(Hd,d))
    print(torch.abs(torch.dot(d,Hd)/torch.norm(Hd)))
    return d , torch.cdiv(Hd,d)
end

function hessianPowermethod2(inputs,targets,param,gradParam, delta, filepath, modelpath) 
    local max_iter = 30
    local epsilon = 10e-8 -- for numerical differentiation to get Hd 
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential(); local model2 = nn.Sequential();
    model:add(dofile(filepath .. modelpath)); model2:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax()); model2:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion(); local criterion2 = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    --local norm_Hd = 1; local norm_Hd_old = 2
    local min_val = 1; local max_val = 2
    local sanity_vector = 0
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    local param_new2,gradParam_eps2 = model2:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    --while (norm_Hd_old - norm_Hd) > delta do
    local i = 0
    while (max_val - min_val) > delta do
        --print("hessianPowermethod")
        --print((norm_Hd_old-norm_Hd))
        param_new:copy(param + d * epsilon)
        param_new2:copy(param - d * epsilon)
        

        --reset gradients
        gradParam_eps:zero()
        gradParam_eps2:zero()

        --feedforward and backpropagation
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do) --gradParams_eps should be updated here 
        local outputs = model2:forward(inputs)
        local f = criterion2:forward(outputs, targets)
        local df_do = criterion2:backward(outputs, targets)
        model2:backward(inputs, df_do) --gradParams_eps2 should be updated here 

        Hd = (gradParam_eps - gradParam_eps2) / 2*epsilon
        norm_Hd_old = norm_Hd
        norm_Hd = torch.norm(Hd)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d = Hd / norm_Hd
        epsilon = 2*torch.sqrt(10e-7)*(1 + torch.norm(param))/torch.norm(d)
        sanity_vector = torch.cdiv(Hd,d)
        max_val = torch.max(sanity_vector) 
        min_val = torch.min(sanity_vector)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
        i = i + 1
        if i > max_iter then break end
    end

    --print("sanity check")
    --print(torch.cdiv(Hd,d))
    return d , torch.cdiv(Hd,d)
end

function negativePowermethod(inputs,targets,param,gradParam, delta, filepath,eigenVal,modelpath) 
    local max_iter = 100 
    local epsilon = 10e-8 -- for numerical differentiation to get Hd
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    --local norm_Md = 1; local norm_Md_old = 2 
    local min_val = 1; local max_val = 2; local min_val2 = 1; local max_val2 = 2;
    local sanity_vector = 0
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    --while (norm_Md_old-norm_Md) > delta do
    local i = 0
    while (max_val - min_val) > delta or (max_val2 - min_val2) > delta  do
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
        sanity_vector = torch.cdiv(Md,d); sanity_vector2 = torch.cdiv(Hd,d);
        max_val = torch.max(sanity_vector); max_val2 = torch.max(sanity_vector2)
        min_val = torch.min(sanity_vector); min_val2 = torch.min(sanity_vector2)
        i = i + 1
        if i > max_iter then break end
    end 

    print("sanity check")
    --print(torch.cdiv(Hd,d))
    --print(torch.cdiv(Md,d))
    
    print(torch.abs(torch.dot(d,Md)/torch.norm(Md)))
    return d , torch.cdiv(Md,d)
end

function negativePowermethod2(inputs,targets,param,gradParam, delta, filepath,eigenVal,modelpath) 
    local max_iter = 100 
    local epsilon = 10e-8 -- for numerical differentiation to get Hd
    -- call the model from src/model 
    -- and put the parameters into this model. 
    local model = nn.Sequential()
    local model2 = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    model2:add(dofile(filepath .. modelpath))
    model2:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()
    local criterion2 = nn.ClassNLLCriterion()
    local d = torch.randn(param:size()) --need to check
    --local norm_Md = 1; local norm_Md_old = 2 
    local min_val = 1; local max_val = 2; local min_val2 = 1; local max_val2 = 2;
    local sanity_vector = 0
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    local param_new2,gradParam_eps2 = model2:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    --while (norm_Md_old-norm_Md) > delta do
    local i = 0
    while (max_val - min_val) > delta or (max_val2 - min_val2) > delta  do
        --print("negativePowermethod")
        --print((norm_Md_old - norm_Md))
        param_new:copy(param + d * epsilon)
        param_new2:copy(param - d * epsilon)

        --reset gradients
        gradParam_eps:zero()
        gradParam_eps2:zero()

        --feedforward and backpropagation
        local outputs = model:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        local outputs = model2:forward(inputs)
        local f = criterion2:forward(outputs, targets)
        local df_do = criterion2:backward(outputs, targets)
        model2:backward(inputs, df_do) --gradParams_eps2 should be updated here 

        local Hd = (gradParam_eps - gradParam_eps2) / 2*epsilon
        Md = d * eigenVal - Hd
        norm_Md_old = norm_Md
        norm_Md = torch.norm(Md)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d = Md / norm_Md
        epsilon = 2*torch.sqrt(10e-7)*(1 + torch.norm(param))/torch.norm(d)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
        sanity_vector = torch.cdiv(Md,d); sanity_vector2 = torch.cdiv(Hd,d);
        max_val = torch.max(sanity_vector); max_val2 = torch.max(sanity_vector2)
        min_val = torch.min(sanity_vector); min_val2 = torch.min(sanity_vector2)
        i = i + 1
        if i > max_iter then break end
    end 

    print("sanity check")
    print(torch.cdiv(Hd,d))
    print(torch.cdiv(Md,d))
    return d , torch.cdiv(Md,d)
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
    --df_do = criterion:backward(outputs, targets)
    --model:backward(inputs, df_do) --gradParams_eps should be updated here 

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
    --df_do = criterion:backward(outputs, targets)
    --model:backward(inputs, df_do) --gradParams_eps should be updated here 

    return loss
end


function checkModelID()
    model = nn.Sequential()
    model:add(dofile('models/train-mnist-model.lua'))
    id = torch.pointer(model)

    
    return id
end

