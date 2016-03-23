a = torch.Tensor(torch.load("powercallRecord.bin"))

size = a:size(1)

b = torch.sum(torch.ge(a,1))

c = torch.sum(torch.ge(a,2))

d = torch.sum(torch.ge(a,3))

print("The number of the case in which...")

print("- a) The norm of gradient is close to zero : " .. b .. "/" .. size)

print("- b) The second test passed (L > M) : " .. c .. "/" .. size)

print("- c) The cost function decreases: " .. d .. "/" .. size)

print("- d) The cost function increases: " .. c - d .. "/" .. size)

