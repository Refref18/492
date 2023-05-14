import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os
import logging
from son_loader import VideoPoseDataset

from son_loader import custom_collate_fn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'working')))
from net.st_gcn import Model
from net.utils.graph_mm import Graph

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\mmpose-full'
info_file="D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\updated\\info.csv"
# Initialize dataset and data loader

# create the train set with new label_dict and index_to_label dictionaries
train_dataset = VideoPoseDataset(data_dir, info_file, train=True)

# create the test set with the same label_dict and index_to_label dictionaries as the train set
test_dataset = VideoPoseDataset(data_dir, info_file,  train=False, label_dict=train_dataset.label_dict, index_to_label=train_dataset.index_to_label)

dataloader_train = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, collate_fn=custom_collate_fn)
dataloader_test = DataLoader(test_dataset, batch_size=4, collate_fn=custom_collate_fn)


print(dataloader_train)

#x-y confidence pose datasının channel sayısı 
#dataloaderdan çek num_classes
model = Model(in_channels=3, num_class=744, graph_args={"layout": "mmpose", "strategy": "spatial"},
              edge_importance_weighting=True, dropout=0.5).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
for epoch in range(10):
    epoch_loss = 0.0
    print(len(dataloader_train))
    for i, inputs in enumerate(dataloader_train):
        #print(len(inputs['face']))
        #print(inputs['label'], inputs['face'])
        labels, inputs = inputs['label'], inputs['face']
        optimizer.zero_grad()
        # Initialize model
        N=4
        in_channels = 3
        T=4
        V=21
        M=80
        outputs = model(torch.Tensor(N, in_channels, T, V, M))
        #outputs = model(inputs.to(device))
        print("OUTPUT",outputs)
        print("Labels",labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(epoch_loss)
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j, inputs in enumerate(dataloader_test):
            labels, inputs = inputs['label'], inputs['face']
            outputs = model(torch.Tensor(N, in_channels, T, V, M))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    logging.info(f'Epoch %d train loss: %.3f, validation loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader_train), val_loss / len(dataloader_test)))

    print('Epoch %d train loss: %.3f, validation loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader_train), val_loss / len(dataloader_test)))



# Save model
#epoch hangi epoch olduğu, 2.key olarak yazdığın şey, 3.sine optimazier state dict(değişen adımlar) 
torch.save(model.state_dict(), 'stgcn_model.pth')

#classes.csv 'yi okuyup filtereleyeceksin.
#scvleri pandasta okursan çok hızlı olur 
#main.py yapman lazım 
#main threadden başlat
