require 'nn'
require 'cunn'

function hessianPowermethodRNN(inputs,targets,param, delta, model) 
    local maxIter = 50
    local acc_threshold = 2
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()
    
    local model_a = model:clone()
    local model_b = model:clone()

    local d = torch.randn(param:size()) --need to check
    d = d / torch.norm(d)
    d = d:cuda()
    
    local d_old = d*10; 
    diff = 10
    local param_new_a, gradParam_eps_a = model_a:getParameters() 
    local param_new_b, gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    local numIters = 0
    while diff > delta and numIters < maxIter do
        numIters = numIters+1
        epsilon = 2*torch.sqrt(10e-8)*(1 + torch.norm(param))/torch.norm(d)

        --print(numIters) --TODO: comment this out
        param_new_a:copy(param + d * epsilon)
        param_new_b:copy(param - d * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()
        


        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        local outputs = model_b:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

        Hd = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        norm_Hd = torch.norm(Hd)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        lambda = torch.dot(d, Hd)
        --print(torch.norm(d-d_old)) --TODO: comment this out
        diff = torch.norm(d*lambda - Hd)
        --print(diff) --TODO: comment this out
        d_old = d
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

function negativePowermethodRNN(inputs,targets,param, delta, maxEigValH, model)
    local maxIter = 50
    local acc_threshold = 2
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()
    
    local model_a = model:clone()
    local model_b = model:clone()

    --model_b:add(nn.LogSoftMax())
    local d = torch.randn(param:size())
    d = d / torch.norm(d)
    d = d:cuda()
    local d_old = d*10; 
    local param_new_a,gradParam_eps_a = model_a:getParameters() 
    local param_new_b,gradParam_eps_b = model_b:getParameters()
    -- in order to reflect loading a new parameter set
    local numIters = 0
    diff_H = 10
    while diff_H > delta and numIters < maxIter do
        collectgarbage() 
        numIters = numIters+1
        epsilon = 2*torch.sqrt(10e-8)*(1 + torch.norm(param))/torch.norm(d)
        --print(numIters) -- TODO: comment this out
        param_new_a:copy(param + d * epsilon)
        param_new_b:copy(param - d * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()
        

        --feedforward and backpropagation
        local outputs = model_a:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        local outputs = model_b:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

        Hd = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        norm_Hd = torch.norm(Hd)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d_old = d
        -- M = mI-H
        Md = d*maxEigValH - Hd
        lambda = torch.dot(d, Md)
        minEigValH = maxEigValH - lambda
        --print(minEigValH)
        --print(torch.norm(d-d_old)) -- TODO: comment this out
        diff_M = torch.norm(d*lambda - Md)
        diff_H = torch.norm(d*minEigValH - Hd)
        --print(diff_M) -- TODO: comment this out
        --print(diff_H) -- TODO: comment this out
        
        d = Md / torch.norm(Md)
        
    end
    converged = false
    if torch.abs(lambda) > diff_M and torch.abs(minEigValH) > diff_H and diff_H < acc_threshold
        then converged = true
    end   
    return minEigValH, d, converged
end


---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function lanczosRNN(inputs,targets,param, delta, model) 
    local maxIter = 50
    local acc_threshold = .2
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()
    
    local model_a = model:clone()
    local model_b = model:clone()

    local param_new_a,gradParam_eps_a = model_a:getParameters() 
    local param_new_b,gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    
    local T = torch.Tensor(maxIter,maxIter):fill(0)-- initialize the tri-diagonal matrix T with zeros
    dim = param:size(1)
    local Vt = torch.CudaTensor(maxIter, dim):fill(0)-- initialize the Krylov matrix with zeros
    -- initialize:
    local v = torch.randn(param:size()) 
    v = v:cuda()
    v = v / torch.norm(v)
    local v_old = 0
    local beta = 0;
    
    for iter =1, maxIter-1 do
        collectgarbage()
        --print(iter) -- TODO: comment this out
        Vt[iter] = v
        
        local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)
        param_new_a:copy(param + v * epsilon)
        param_new_b:copy(param - v * epsilon)

        --reset gradients
        gradParam_eps_a:zero()
        gradParam_eps_b:zero()
        

        --feedforward and backpropagation--       
        local outputs = model_a:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_a:backward(inputs, df_do) --gradParams_eps should be updated here 
        
        local outputs = model_b:forward(inputs)
        --local outputs_b = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 
 

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
        T[iter][iter+1] = beta
        T[iter+1][iter] = beta 
    end
    collectgarbage()
    Vt[maxIter] = v
    local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)
    param_new_a:copy(param + v * epsilon)
    param_new_b:copy(param - v * epsilon)

    --reset gradients
    gradParam_eps_a:zero()
    gradParam_eps_b:zero()

    

    --feedforward and backpropagation
    local outputs = model_a:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    local outputs = model_b:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local w = Hv 
    local alpha = torch.dot(w,v)
    T[maxIter][maxIter] = alpha -- will get an error if not corret
    
    -- find eigenvalues and eigenvectors of T
    lambdas, V_T = torch.symeig(T, 'V') -- maximal eigenvalue of T and the corresponding eigenvector
    local yy, i = torch.sort(torch.abs(lambdas))
    lambda = yy[maxIter]
    V_TT = V_T:t()
    v = V_TT[i[maxIter]]
    --------------------------------------------------
    -- get eigenvector of H
    V = Vt:t() -- now V is the Krylov matrix
    v = v:cuda()
    v = torch.mv(V, v)
    v = v / torch.norm(v)
    --compute Hv
    param_new_a:copy(param + v * epsilon)
    param_new_b:copy(param - v * epsilon)

    --reset gradients
    gradParam_eps_a:zero()
    gradParam_eps_b:zero()

    epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)

    --feedforward and backpropagation
    local outputs = model_a:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    local outputs = model_b:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local diff = torch.norm(Hv - v*lambda)
    --print(diff) -- TODO: comment this out
    converged = torch.abs(lambda) > diff and diff<acc_threshold
    return lambda, v, converged
end



---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function negativeLanczosRNN(inputs,targets,param, delta, maxEigValH, model)
    local maxIter = 50
    local acc_threshold = .2
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()

    local model_a = model:clone()
    local model_b = model:clone()
    
    local param_new_a,gradParam_eps_a = model_a:getParameters() 
    local param_new_b,gradParam_eps_b = model_b:getParameters() 
    -- in order to reflect loading a new parameter set
    
    local T = torch.Tensor(maxIter,maxIter):fill(0)-- initialize the tri-diagonal matrix T with zeros
    dim = param:size(1)
    local Vt = torch.CudaTensor(maxIter, dim):fill(0)-- initialize the Krylov matrix with zeros
    -- initialize:
    local v = torch.randn(param:size())
    v = v:cuda()  
    v = v / torch.norm(v)
    local v_old = 0
    local beta = 0;
    
    for iter =1, maxIter-1 do
        collectgarbage()
        --print(iter) -- TODO: comment this out
        Vt[iter] = v
        
        local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)

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
        local outputs = model_b:forward(inputs)
        local f = criterion:forward(outputs, targets)
        local df_do = criterion:backward(outputs, targets)
        model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

        local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
        -- M = mI-H
        local Mv = v*maxEigValH - Hv
        Hv = nil
        local ww = Mv -- omega prime
        local alpha = torch.dot(ww, v)
        local w = ww - v*alpha - v_old*beta
        ww = nil
        Mv = nil
        beta = torch.norm(w)
        v_old = v:clone()
        v = w / beta
        w = nil
        T[iter][iter] = alpha
        T[iter][iter+1] = beta
        T[iter+1][iter] = beta 
    end
    collectgarbage()
    Vt[maxIter] = v
    local epsilon = 2*torch.sqrt(1e-7)*(1 + torch.norm(param))/torch.norm(v)
    param_new_a:copy(param + v * epsilon)
    param_new_b:copy(param - v * epsilon)

    --reset gradients
    gradParam_eps_a:zero()
    gradParam_eps_b:zero()



    --feedforward and backpropagation
    local outputs = model_a:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    local outputs = model_b:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local Mv = v*maxEigValH - Hv
    
    local w = Mv 
    local alpha = torch.dot(w,v)
    T[maxIter][maxIter] = alpha -- will get an error if not corret
    
    -- find eigenvalues and eigenvectors of T
    lambdas, V_T = torch.symeig(T, 'V') -- maximal eigenvalue of T and the corresponding eigenvector
    local yy, i = torch.sort(torch.abs(lambdas))
    lambda = yy[maxIter]
    V_TT = V_T:t()
    v = V_TT[i[maxIter]]
    --------------------------------------------------
    -- get eigenvector of H
    V = Vt:t() -- now V is the Krylov matrix
    v = v:cuda()
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
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_a:backward(inputs, df_do) --gradParams_eps should be updated here 

    local outputs = model_b:forward(inputs)
    --local outputs_b = model_b:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    model_b:backward(inputs, df_do) --gradParams_eps should be updated here 

    local Hv = (gradParam_eps_a - gradParam_eps_b) / (2*epsilon)
    local Mv = v*maxEigValH - Hv
    
    minEigValH = maxEigValH - lambda
    local diff_M = torch.norm(v*lambda - Mv)
    local diff_H = torch.norm(v*minEigValH - Hv)
    
    --print(diff_M) -- TODO: comment this out
    --print(diff_H) -- TODO: comment this out
    converged = torch.abs(minEigValH) > diff_H and torch.abs(lambda) > diff_M and diff_H < acc_threshold
    return minEigValH, v, converged
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function computeLineSearchLossRNN(inputs,targets,parameters,model,eigenVector,stepSize)
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()
    
    local model_a = model:clone()
    local param_new,gradParam_eps = model_a:getParameters() --I need to do this
    param_new:copy(parameters)
    param_new:add(eigenVector*stepSize)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)
    return loss
end

---------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------


function computeCurrentLossRNN(inputs,targets,parameters,model)
    local crit = nn.ClassNLLCriterion()
    local criterion = nn.SequencerCriterion(crit)
    criterion:cuda()
    
    local model_a = model:clone()

    local param_new,gradParam_eps = model_a:getParameters() --I need to do this
    param_new:copy(parameters)

    outputs = model:forward(inputs)
    loss = criterion:forward(outputs, targets)

    return loss
end
