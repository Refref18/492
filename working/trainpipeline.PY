import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VideoPoseDataset
from torch.nn.functional import mse_loss
from dataloader import custom_collate_fn
from net.st_gcn import st_gcn
from net.utils.graph_mm import Graph
# Define the model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize dataset and data loader
data_dir = 'D:/2022-2023 2.dönem/Bitirme Projesi/face'
dataset = VideoPoseDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, collate_fn=custom_collate_fn)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
num_nodes = 18
in_channels = 2
num_classes = 10
layout = 'openpose'
kernel_size = (1, 3)
stride = (1, 1)
dropout = 0.5
graph_args = {
    'layout': layout,
    'strategy': 'spatial'
}


in_channels = 3
out_channels = 64
kernel_size = (9, 5)
stride = 1
dropout = 0.5
residual = True


# Create an instance of the STGCN model
model = st_gcn(in_channels=in_channels,
               out_channels=out_channels,
               kernel_size=kernel_size,
               stride=stride,
               dropout=dropout,
               residual=residual
               ).to(device)
# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs=10
# loop over the epochs
for epoch in range(num_epochs):

    # loop over the batches in the dataloader
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        # move the inputs and targets to the device
        inputs = inputs.to(device)
        targets = targets.to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate the loss
        loss = mse_loss(outputs, targets)

        # backward pass
        loss.backward()

        # update the weights
        optimizer.step()

        # print the loss every 10 batches
        if batch_idx % 10 == 0:
            print('Epoch: {} Batch: {} Loss: {}'.format(
                epoch, batch_idx, loss.item()))
    
# Save model
torch.save(model.state_dict(), 'stgcn_model.pth')
        