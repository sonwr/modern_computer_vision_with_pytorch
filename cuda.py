import torch
x = torch.rand(1, 300)
y = torch.rand(300, 200)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device == 'cuda', "This exercise assumes the notebook is on a GPU machine"

x, y = x.to(device), y.to(device)
z=(x@y)

print(z)