import torch
import numpy as np

# init a tensor from array
data = [[1, 2], [3, 4]]
data_tensor = torch.tensor(data)

# init a tensor from NumPy array
np_array = np.array(data)
data_tensor_np_array = torch.tensor(np_array)

# init a tensor from another one
x_ones = torch.ones_like(data_tensor)
# print(f"Ones Tensor:\n {x_ones}\n")
x_rand = torch.rand_like(data_tensor, dtype=torch.float)
# print(f"Rand Tensor:\n {x_rand}\n")

# shape is a tuple of tensor dimensions
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape, dtype=torch.int)
zeros_tensor = torch.zeros(shape, dtype=torch.double)
# print(f"Random Tensor: \n {rand_tensor} \n")
# print(f"Ones Tensor: \n {ones_tensor} \n")
# print(f"Zeros Tensor: \n {zeros_tensor}")

# check tensor attributes
tensor = torch.tensor(data, dtype=torch.float).to("cuda")
# But copying large tensors across devices can be expensive
# in terms of time and memory!
print(f"Tensor: {tensor}")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])  # or tensor[:, -1]
# tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# matrix multiply
y1 = tensor @ tensor.T
print(y1)
y2 = tensor.matmul(tensor.T)
print(y2)
y3 = torch.matmul(tensor, tensor.T)
print(y3)

# dot multiply
z1 = tensor * tensor
print(z1)
z2 = tensor.mul(tensor)
print(z2)

agg = tensor.sum()
print(agg)  # this still is a tensor
agg_item = agg.item()
print(agg_item)  # now it's a Python numerical value

# convert a tensor to NumPy array
t = torch.ones(5)
print(f"t: {t} ")
t.add_(2)
n = t.numpy()  # n = np.ones(5)
print(f"n: {n} ")
t2 = torch.from_numpy(n)
print(f"t2 {t2}")
