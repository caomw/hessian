require 'nn'
function sparse_init(model) 
    modules = model:findModules('nn.Linear')
    for i = 1, #modules do
        local weight_matrix = torch.rand(modules[i].weight:size())
        local n = weight_matrix:size(1) 
        local m = weight_matrix:size(2) --following the convention s.t. n x m matrix
        local k = 15 -- the number of connecting weights
        for row = 1, n do -- iterate temp over row-wise to fill 15 of weights for each row. 
            random_index = torch.randperm(weight_matrix:size(2)) -- generate random permutation of a sequence from 1 to (size of row)
            extracted_index = random_index[{{1,m-k}}] -- extrac the first m-15 numbers. Will replace them with zero
            byte_vector = torch.zeros(m-k,m):scatter(2, extracted_index:long():view(m-k,1), 1):sum(1):byte() -- among m's numbers, m-15's number is 1.
            weight_matrix[row][byte_vector] = 0 
        end
        modules[i].weight = weight_matrix --I think even without this the modification is already done because I didn't clone it, but just changed the pointer. 
    end
    return model
end



