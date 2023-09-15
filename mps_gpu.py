import torch
x = torch.rand(1, 300)
y = torch.rand(300, 200)

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else: 
    device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')

    x, y = x.to(device), y.to(device)
    z=(x@y)

    print(z)