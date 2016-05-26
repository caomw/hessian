require 'nn'
function hessianPowermethodClassifier(inputs, targets, param, delta, filepath, modelpath) 
    local maxIter = 20
    local acc_threshold = .2
    local criterion = nn.ClassNLLCriterion()
    --criterion.sizeAverage = false
    local model_a  = nn.Sequential ()                                                                                                                                            
    --model_a:add(dofile(modelFile))
    model_a:add(dofile(filepath .. modelpath))
    model_a:add(nn.LogSoftMax())

    local model_b  = nn.Sequential ()                                                                                                                                            
    --model_b:add(dofile(modelFile))
    model_b:add(dofile(filepath .. modelpath))
    model_b:add(nn.LogSoftMax())

    local d = torch.randn(param:size())
    d = d / torch.norm(d)
    
    local diff = 10
    local param_new_a, gradParam_eps_a = model_a:getParameters() 
    local param_new_b, gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    local numIters = 0
    while diff > delta and numIters < maxIter do
        numIters = numIters+1
        epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(d)

        --print(numIters) --TODO: comment this out
        param_new_a:copy(param + d * epsilon)
        param_new_b:copy(param - d * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()

        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
             
        f = f/inputs:size(1)
        gradParam_eps_a:div(inputs:size(1))

        local outputs = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        f = f/inputs:size(1)
        gradParam_eps_b:div(inputs:size(1))
  

        local Hd = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        local norm_Hd = torch.norm(Hd)
        lambda = torch.dot(d, Hd)
        diff = torch.norm(d*lambda - Hd) --TODO: comment this out
        --print('|Hv-lambda v|: '..diff) --TODO: comment this out
        --print('lambda: '..lambda) --TODO: comment this out
        d = Hd / norm_Hd
    end
    converged = false
    if torch.abs(lambda) > diff and diff < acc_threshold
        then converged = true
    end
    return lambda, d, converged
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

function negativePowermethodClassifier(inputs, targets, param, delta, filepath, modelpath, maxEigValH) 
    local maxIter = 40
    local acc_threshold = .2
    local criterion = nn.ClassNLLCriterion()
    --criterion.sizeAverage = false
    local model_a  = nn.Sequential ()                                                                                                                                            
    --model_a:add(dofile(modelFile))
    model_a:add(dofile(filepath .. modelpath))
    model_a:add(nn.LogSoftMax())

    local model_b  = nn.Sequential ()                                                                                                                                            
    --model_b:add(dofile(modelFile))
    model_b:add(dofile(filepath .. modelpath))
    model_b:add(nn.LogSoftMax())

    local d = torch.randn(param:size()) --need to check
    d = d / torch.norm(d)
    
    local diff = 10
    local param_new_a, gradParam_eps_a = model_a:getParameters() 
    local param_new_b, gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    local numIters = 0
    while diff > delta and numIters < maxIter do
        numIters = numIters+1
        epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(d)

        param_new_a:copy(param + d * epsilon)
        param_new_b:copy(param - d * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()

        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
             
        f = f/inputs:size(1)
        gradParam_eps_a:div(inputs:size(1))

        local outputs = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        f = f/inputs:size(1)
        gradParam_eps_b:div(inputs:size(1))
  

        local Hd = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        local norm_Hd = torch.norm(Hd)

        local Md = d*maxEigValH - Hd
        local lambda = torch.dot(d, Md)
        minEigValH = maxEigValH - lambda

        local diff_M = torch.norm(d*lambda - Md)
        diff_H = torch.norm(d*minEigValH - Hd)
        --print('|Hv-lambda v|: '..diff_H) --TODO: comment this out
        --print('lambda: '..minEigValH) --TODO: comment this out
        d = Md / torch.norm(Md)
    end
    converged = false
    if torch.abs(minEigValH) > diff_H and diff_H < acc_threshold
        then converged = true
    end
    return minEigValH, d, converged
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

function lanczosClassifier(inputs, targets, param, delta, filepath, modelpath) 
    local maxIter = 20
    local acc_threshold = 2
    local criterion = nn.ClassNLLCriterion()
    --criterion.sizeAverage = false
    local model_a  = nn.Sequential ()                                                                                                                                            
    --model_a:add(dofile(modelFile))
    model_a:add(dofile(filepath .. modelpath))
    model_a:add(nn.LogSoftMax())

    local model_b  = nn.Sequential ()                                                                                                                                            
    --model_b:add(dofile(modelFile))
    model_b:add(dofile(filepath .. modelpath))
    model_b:add(nn.LogSoftMax())

    local param_new_a, gradParam_eps_a = model_a:getParameters() 
    local param_new_b, gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    
    local T = torch.Tensor(maxIter,maxIter):fill(0)-- initialize the tri-diagonal matrix T with zeros
    dim = param:size(1)
    local Vt = torch.Tensor(maxIter, dim):fill(0)-- initialize the Krylov matrix with zeros
    -- initialize:
    local v = torch.randn(param:size()) 

    v = v/torch.norm(v)
    local v_old = 0
    local beta = 0;
    
    for iter =1, maxIter do
        Vt[iter] = v
        
        epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)
        param_new_a:copy(param + v * epsilon)
        param_new_b:copy(param - v * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()
        

        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
  
        f = f/inputs:size(1)
        gradParam_eps_a:div(inputs:size(1))
        
        local outputs = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 
    
        f = f/inputs:size(1)
        gradParam_eps_b:div(inputs:size(1))

        local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        local ww = Hv -- omega prime
        local alpha = torch.dot(ww, v)
        local w = ww - v*alpha - v_old*beta
        ww = nil 
        Hv = nil
        beta = torch.norm(w)
        v_old = v:clone()
        v = w / beta
        w = nil
        T[iter][iter] = alpha
        if iter<maxIter then
            T[iter][iter+1] = beta
            T[iter+1][iter] = beta 
        end    
    end

    -- find eigenvalues and eigenvectors of T
    lambdas, V_T = torch.symeig(T, 'V') -- maximal eigenvalue of T and the corresponding eigenvector
    local yy, i = torch.sort(torch.abs(lambdas))
    lambda = yy[maxIter]
    V_TT = V_T:t()
    v = V_TT[i[maxIter]]
    --------------------------------------------------
    -- get eigenvector of H
    V = Vt:t() -- now V is the Krylov matrix
    v = torch.mv(V, v)
    v = v / torch.norm(v)
    --compute Hv
    param_new_a:copy(param + v * epsilon)
    param_new_b:copy(param - v * epsilon)

    --reset gradients
    gradParam_eps_a:zero()
    gradParam_eps_b:zero()

    local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)

    --feedforward and backpropagation
    local outputs = model_a:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    f = f/inputs:size(1)
    gradParam_eps_a:div(inputs:size(1))

    local outputs = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    f = f/inputs:size(1)
    gradParam_eps_b:div(inputs:size(1))

    local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local diff = torch.norm(Hv - v*lambda)
    --print('|Hv-lambda v|: '..diff) -- TODO: comment this out
    converged = torch.abs(lambda) > diff and diff<acc_threshold
    return lambda, v, converged
end
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

function negativeLanczosClassifier(inputs, targets, param, delta, filepath, modelpath, maxEigValH) 
    local maxIter = 40
    local acc_threshold = 2
    local criterion = nn.ClassNLLCriterion()
    --criterion.sizeAverage = false
    local model_a  = nn.Sequential ()                                                                                                                                            
    --model_a:add(dofile(modelFile))
    model_a:add(dofile(filepath .. modelpath))
    model_a:add(nn.LogSoftMax())

    local model_b  = nn.Sequential ()                                                                                                                                            
    --model_b:add(dofile(modelFile))
    model_b:add(dofile(filepath .. modelpath))
    model_b:add(nn.LogSoftMax())

    local param_new_a, gradParam_eps_a = model_a:getParameters() 
    local param_new_b, gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    
    local T = torch.Tensor(maxIter,maxIter):fill(0)-- initialize the tri-diagonal matrix T with zeros
    dim = param:size(1)
    local Vt = torch.Tensor(maxIter, dim):fill(0)-- initialize the Krylov matrix with zeros
    -- initialize:
    local v = torch.randn(param:size()) 
    v = v/torch.norm(v)
    local v_old = 0
    local beta = 0;
    
    for iter =1, maxIter do
        Vt[iter] = v
        
        epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)
        param_new_a:copy(param + v * epsilon)
        param_new_b:copy(param - v * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()
        

        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
  
        f = f/inputs:size(1)
        gradParam_eps_a:div(inputs:size(1))
        
        local outputs = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 
    
        f = f/inputs:size(1)
        gradParam_eps_b:div(inputs:size(1))

        local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        local Mv = v*maxEigValH - Hv
        Hv = nil
        local ww = Mv -- omega prime
        local alpha = torch.dot(ww, v)
        local w = ww - v*alpha - v_old*beta
        ww = nil 
        Hv = nil
        beta = torch.norm(w)
        v_old = v:clone()
        v = w / beta
        w = nil
        T[iter][iter] = alpha
        if iter<maxIter then
            T[iter][iter+1] = beta
            T[iter+1][iter] = beta 
        end    
    end

    -- find eigenvalues and eigenvectors of T
    lambdas, V_T = torch.symeig(T, 'V') -- maximal eigenvalue of T and the corresponding eigenvector
    local yy, i = torch.sort(torch.abs(lambdas))
    lambda = yy[maxIter]
    V_TT = V_T:t()
    v = V_TT[i[maxIter]]
    --------------------------------------------------
    -- get eigenvector of H
    V = Vt:t() -- now V is the Krylov matrix
    v = torch.mv(V, v)
    v = v / torch.norm(v)
    --compute Hv
    param_new_a:copy(param + v * epsilon)
    param_new_b:copy(param - v * epsilon)

    --reset gradients
    gradParam_eps_a:zero()
    gradParam_eps_b:zero()

    local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)

    --feedforward and backpropagation
    local outputs = model_a:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    f = f/inputs:size(1)
    gradParam_eps_a:div(inputs:size(1))

    local outputs = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    f = f/inputs:size(1)
    gradParam_eps_b:div(inputs:size(1))

    local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local Mv = v*maxEigValH - Hv
    
    minEigValH = maxEigValH - lambda
    local diff_M = torch.norm(v*lambda - Mv)
    local diff_H = torch.norm(v*minEigValH - Hv)
    
    --print(diff_M) -- TODO: comment this out
    --print('|Hv-lambda v|: '..diff_H) -- TODO: comment this out
    converged = torch.abs(minEigValH) > diff_H and torch.abs(lambda) > diff_M and diff_H < acc_threshold
    return minEigValH, v, converged
end
---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


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
    return loss
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function computeCurrentLoss(inputs,targets,parameters,filepath,modelpath)
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    model:add(nn.LogSoftMax())
    local criterion = nn.ClassNLLCriterion()

    local param_new,gradParam_eps = model:getParameters() --I need to do this
    param_new:copy(parameters)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)

    return loss
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function computeLineSearchLossAE(inputs,targets,parameters,filepath,modelpath,eigenVector,stepSize)
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    --model:add(nn.LogSoftMax())
    local criterion = nn.MSECriterion()
    --criterion.sizeAverage = false

    local param_new,gradParam_eps = model:getParameters() --I need to do this
    param_new:copy(parameters)
    param_new:add(eigenVector*stepSize)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)
    return loss
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function computeCurrentLossAE(inputs,targets,parameters,filepath,modelpath)
    local model = nn.Sequential()
    model:add(dofile(filepath .. modelpath))
    --model:add(nn.LogSoftMax())
    local criterion = nn.MSECriterion()
    --criterion.sizeAverage = false
    local param_new,gradParam_eps = model:getParameters() --I need to do this
    param_new:copy(parameters)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)

    return loss
end
