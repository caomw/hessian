require 'nn'
function negativePowermethod(inputs,targets,param,gradParam, eps, filepath,eigen) 
    epsilon = 10e-8 -- for numerical differentiation to get Hd
    -- call the model from src/model 
    -- and put the parameters into this model. 
    model = nn.Sequential()
    model:add(dofile(filepath .. '/models/train-mnist-model.lua'))
    model:add(nn.LogSoftMax())
    criterion = nn.ClassNLLCriterion()
    d = torch.randn(param:size()) --need to check
    d_old = d + 1
    param_new,gradParam_eps = model:getParameters() --I need to do this
    -- in order to reflect loading a new parameter set
    while torch.norm(d_old - d) > 10e-6 do
        param_new = param + d * epsilon

        --reset gradients
        gradParam_eps:zero()

        --feedforward and backpropagation
        outputs = model:forward(inputs)
        f = criterion:forward(outputs, targets)
        df_do = criterion:backward(outputs, targets)
        model:backward(inputs, df_do) --gradParams_eps should be updated here 

        Hd = (gradParam_eps - gradParam) / epsilon
        Md = d * eigenVal - Hd
        norm_Md = torch.norm(Md)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        d_old = d
        d = Md / norm_Md
        epsilon = 2*torch.sqrt(10e-7)*(1 + torch.norm(param))/torch.norm(d)
        --print(norm_Hd)--this norm converges to the dominant eigenvalue 
    end

    print("sanity check")
    print(torch.cdiv(Md,d))
    return d , torch.cdiv(Md,d)[1] 
end


