import torch

x = torch.ones(5)
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
print(z.requires_grad)
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# sometimes wo do not want to support gradient computation
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)
# or use detach() method to achieve same goal
z_det = z.detach()
print(z_det.requires_grad)

loss.backward(retain_graph=True)  # set retain_graph to True if we need to do several backward
print(w.grad)
print(b.grad)

inp = torch.eye(2, requires_grad=True)
out = (inp+1).pow(2)
print(inp)
print(out)
# grad can be implicitly created only for scalar outputs, but y is a tensor
# by introducing torch.ones_like(input), warp a function to y, and grad can ce calculated
out.backward(torch.ones_like(inp), retain_graph=True)
print("First Call\n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("Second Call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("Call after clean\n", inp.grad)

