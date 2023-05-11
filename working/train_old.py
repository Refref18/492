import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloader import VideoPoseDataset

from dataloader import custom_collate_fn
from net.st_gcn import Model
from net.utils.graph_mm import Graph

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and data loader
data_dir = 'D:/2022-2023 2.dönem/Bitirme Projesi/face'
dataset = VideoPoseDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, collate_fn=custom_collate_fn)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)




model = Model(in_channels=3, num_class=6, graph_args={"layout": "mmpose", "strategy": "spatial"},
              edge_importance_weighting=True, dropout=0.5).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(10):
    epoch_loss = 0.0
    for i, (inputs) in enumerate(dataloader):
        print(inputs['label'], inputs['face'])
        labels, inputs = inputs['label'], inputs['face']
        optimizer.zero_grad()
        # Initialize model
        graph = Graph(layout='mmpose', strategy='uniform',
                    max_hop=1, dilation=1)
        # Access the adjacency matrix
        adj_matrix = graph.A
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader)))

# Save model
torch.save(model.state_dict(), 'stgcn_model.pth')
