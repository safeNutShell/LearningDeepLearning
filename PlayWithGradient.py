import torch

# create a tensor and track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

# do some operations
# since y is a result of an operation, so it has a grad_fn which reference a 'Function'
y = x + 2
print(y)
print(y.grad_fn)

# more operations!
z = y * y * 3
out = z.mean()
print(z, out)

# The require_grad flag is false in default
a = torch.randn(2, 3)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)

# let's do backprop
# since out is a scalar, bacjward() dont need argument
out.backward()
print(x.grad)

# we can use autograd to carry vector-jacobian product
x2 = torch.randn(3, 3, requires_grad=True)
y2 = x2 * x2
print(x2)
print(y2)
while y2.data.norm() < 1000:
    y2 = y2 * 2
# since y2 is a matrix, we cannot directly call backward()
# so we need convert y2 to scalar, which doule be achieved by dot-product with a I-matrix
w = torch.ones_like(x2)
print(w)
y2.backward(w)
print(x2.grad)

# whitin torch.no_grad() block, autograd is stopped(on new variable)
print(x2.requires_grad)
with torch.no_grad():
    print((x2 ** 2).requires_grad)
print(x2.requires_grad)

# or use detach() to get a copy but requires_grad=False
x3 = x2.detach()
print(x3.requires_grad)
