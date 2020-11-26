import torch
import numpy

# create a uninitialized matrix
m1 = torch.empty(5, 3)
print(m1)

# create a randomly initialized matrix
m2 = torch.rand(5, 3)
print(m2)

# create a matrix filled by zeros
m3 = torch.zeros(5, 3, dtype=torch.long)
print(m3)

# create a tensor given data
m4 = torch.tensor([[5.3, 7], [3.5, 9.2]])
print(m4)

# create a tensor based on existing tensor i.e. some properties like dtype
m5 = m4.new_ones(2, 3, dtype=torch.double)
print(m5)

# another way to reuse existing tensor but filled with random values
m6 = torch.randn_like(m5, dtype=torch.float)
print(m6)

# just print size
print(m6.size())

# carry addition operation
m2_plus_m3 = torch.add(m2, m3)
"""
this format is equal
res = torch.empty(5, 3)
torch.add(m2, m2, out=res)
"""
print(m2_plus_m3)

# addition in-place
# any operation with post-fixed '_' is in-place
x_add_in_place = torch.zeros(2, 3, dtype=torch.float)
y_add_in_place = torch.rand(2, 3)
print(y_add_in_place)
x_add_in_place.add_(y_add_in_place)
print(x_add_in_place)

# tensor can be resized while the values are retained
# m2.view(-1, 5) means the first dimension is inferred from others
m2_resized = m2.view(15)
print(m2.size(), m2_resized)
print(m2_resized)

# torch tensor and numpy array can convert to each other
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# a and b are behave like pointers
a.add_(1)
print(a)
print(b)

# convert numpy array to tensor
c = numpy.ones(5)
d = torch.from_numpy(c)
numpy.add(c, 1, out=c)
print(c)
print(d)

# sadly, i have no cuda otherwise the operation can be accelerated
device = torch.device("cpu")
print(device)
device_test = torch.ones(2, 2, device=device)
print(device_test)
