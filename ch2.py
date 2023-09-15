import torch
x = torch.rand(5, 3)
print(x)


import torch
x = torch.tensor([[1,2]])
y = torch.tensor([[1],[2]])
print(x.shape)
# torch.Size([1,2]) # one entity of two items
print(y.shape)
# torch.Size([2,1]) # two entities of one item each
print(x.dtype)
# torch.int64

x = torch.tensor([False, 1, 2.0])
print(x)
# tensor([0., 1., 2.])


y = torch.tensor([2, 3, 1, 0]) # y.shape == (4)
print(y)
y = y.view(4,1)                # y.shape == (4, 1)
print(y)
y = y.view(1,4)
print(y)



x = torch.randn(10,1,10)
z1 = torch.squeeze(x, 1) # similar to np.squeeze()
# The same operation can be directly performed on
# x by calling squeeze and the dimension to squeeze out
z2 = x.squeeze(1)
assert torch.all(z1 == z2) # all the elements in both tensors are equal
print('Squeeze:\n', x.shape, z1.shape)
print(z1.shape)
print(z2.shape)

z = torch.randint(low=0, high=10, size=(2,3))
print(z)



x = torch.rand(2,2,2);
print (x);
print(x.max())
print(x.max(dim=0))

print("--")
x = torch.arange(25).reshape(5, 5)
print(x)


x = torch.tensor([[2., -1.], [1., 1.]], requires_grad=True)
y = torch.tensor([[2., -1.], [1., 1.]])
print(x)
print(y)

out = x.pow(2).sum()
print(out)
out.backward()

print(x)
print(out)
