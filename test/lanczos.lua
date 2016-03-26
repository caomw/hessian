function lanczos(A,k)
    n = A:size(1)
    V = torch.Tensor(n,k+1) -- n x k+1
    Vt = V:t() -- (k+1) x n
    Vt[1] = 0 -- v0 = 0
    b = torch.Tensor(1,k):fill(0):t() -- row vector
    a = torch.Tensor(1,k):fill(0):t() -- row vector
    b[1] = 0 -- b1 = 0
    v1 = torch.randn(n)      
    v1_norm = torch.norm(v1)
    v1 = v1 / v1_norm
    Vt[2] = v1 -- v1 = random vector with norm 1
    for j = 1, k-1 do
        print(Vt[j+1])
        w_t = A*Vt[j+1] -- A * v_j; w_t = column vector (in Torch)
        a[j] = Vt[j+1]:dot(w_t) -- column vector (w_t) * row vector (Vt[j]) in Torch
        print(w_t)
        print(Vt[j+1])
        print(a[j])
        w = w_t - a[j][1]*Vt[j+1] - b[j][1]*Vt[j] -- w = column (in Torch)
        b[j+1] = torch.norm(w)
        Vt[j+2] = (w/b[j+1])
    end
    w_m = A * (Vt[k+1])
    a[k] = Vt[k+1] * w_m

    -- construct T
    T = torch.Tensor(k,k):fill(0)
    for i=1,k do
        for j=1,k do
            if i == j then
                T[i][j] = a[i]
                if j+1 <= k then
                    T[i][j+1] = b[j+1]
                end
                if 0 <= j-1 then
                    T[i][j-1] = b[j-1]
                end
            end
        end
    end

    return T
end

