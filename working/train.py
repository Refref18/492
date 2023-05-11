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
data_dir = 'C:\\Users\\eatron1\\Desktop\\termproject-main\\termproject-main\\working'
dataset = VideoPoseDataset(data_dir)

#train dataset ve test dataset iki tane farklı dataset loader 
#test train diye 2 şeye gerek yok
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, collate_fn=custom_collate_fn)
#testte shuffle etme

# dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
print(dataloader)
#x-y confidence pose datasının channel sayısı 
#dataloaderdan çek num_classes
model = Model(in_channels=3, num_class=6, graph_args={"layout": "mmpose", "strategy": "spatial"},
              edge_importance_weighting=True, dropout=0.5).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)#adam w

# Train model
for epoch in range(10):
    epoch_loss = 0.0
    for i, (inputs) in enumerate(dataloader):
        print(inputs['label'], inputs['face'])
        labels, inputs = inputs['label'], inputs['face']
        optimizer.zero_grad()
        # Initialize model
        """graph = Graph(layout='mmpose', strategy='uniform',
                    max_hop=1, dilation=1)
        # Access the adjacency matrix
        adj_matrix = graph.A"""
        """
        N is the batch size
        in_channels is the number of channels in the input data
        T is the length of the input sequence
        V is the number of graph nodes
        M is the number of instances in a frame
        """
        N=1
        in_channels = 3
        T=4
        V=21
        M=80
        outputs = model(torch.Tensor(N, in_channels, T, V, M))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        # train testi bir şekilde eklemen lazım epoch dataloader yeni bir loop train ve test olan 
        """traşn ederken trainin dataloaderından geçircez
        lost baskwardd çalışcak
        o loopa ikinci kere girdiği zaman(test) BACKWARD YAPMA!! sadece foward""" 

        #foward backward
    print('Epoch %d loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader)))
    #text doyasına yaz logging farklı bir streame yönlendir EN SON 
    #fprintf gibi de olur , csv de olur satır satır, validation loss, learning rate, model ne predict etmiş 




# Save model
#epoch hangi epoch olduğu, 2.key olarak yazdığın şey, 3.sine optimazier state dict(değişen adımlar) 
torch.save(model.state_dict(), 'stgcn_model.pth')

#classes.csv 'yi okuyup filtereleyeceksin.
#scvleri pandasta okursan çok hızlı olur 
#main.py yapman lazım 
#main threadden başlat
