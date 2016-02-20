function powermethod(A, b, eps, maxiter) -- either eps or iter is used 
    for i = 1 , 10 do
        temp = A*b
        norm = torch.norm(temp)
        -- normalize the resultant vector to a unit vector
        -- for the next iteration
        b = temp / norm
        print(norm)
    end
    return norm --this norm converges to the dominant eigenvalue 
end


