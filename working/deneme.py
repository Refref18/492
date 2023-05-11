import torch

# create a tensor from a list of values
inputs = torch.tensor([1, 2, 3])
labels = torch.tensor([0, 1, 0])

# move the tensors to a GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = inputs.to(device)
labels = labels.to(device)

print(labels)
