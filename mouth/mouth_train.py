import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os
import logging
from mouth_loader import VideoPoseDataset
import torch.optim.lr_scheduler as lr_scheduler
import csv
from mouth_loader import custom_collate_fn
from net.st_gcn_mouth import Model
from net.utils.mouth_graph import Graph

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_dir = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\mmpose-full'
info_file="D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\updated\\info.csv"
# Initialize dataset and data loader
graph = Graph(**{"layout": "mmpose_mouth_11", "strategy": "spatial"})
# create the train set with new label_dict and index_to_label dictionaries
train_dataset = VideoPoseDataset(data_dir, info_file, train=True,unique_nodes=graph.unique_nodes)

# create the test set with the same label_dict and index_to_label dictionaries as the train set
test_dataset = VideoPoseDataset(data_dir, info_file,  train=False, label_dict=train_dataset.label_dict, index_to_label=train_dataset.index_to_label,unique_nodes=graph.unique_nodes)

dataloader_train = DataLoader(train_dataset, batch_size=4,
                        shuffle=True, collate_fn=custom_collate_fn)
dataloader_test = DataLoader(test_dataset, batch_size=4, collate_fn=custom_collate_fn)





#x-y confidence pose datasının channel sayısı 
#dataloaderdan çek num_classes

model = Model( in_channels=3, num_class=744, graph=graph,
              edge_importance_weighting=True).to(device)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()#binary!
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Define your learning rate scheduler
milestones = [25, 45]  # Epochs at which to decay the learning rate
decay_factor = 0.1  # Factor by which to decay the learning rate
scheduler = lr_scheduler.MultiStepLR(
    optimizer, milestones=milestones, gamma=decay_factor)
print(len(dataloader_train))
# Open CSV file in write mode

for epoch in range(60):
    epoch_loss = 0.0
    #print(len(dataloader_train))
    model.train()
    for i,a in enumerate(dataloader_train):
        print(i)
        img = a['face']
        img = torch.tensor(img, dtype=torch.float32)
        #print(img)
        labels=a['label']
            
        print("LABELS :",labels )
        #print(img,labels)
        #print(len(inputs['face']))
        #print(inputs['label'], inputs['face'])
        # labels, inputs = inputs['label'], inputs['face']
        optimizer.zero_grad()
        # PYCM CONFUSION
        # cborn - heat map dpi artır
        # Initialize model
        outputs = model(img)
        print("outputs ",outputs)
        #outputs = model(inputs.to(device))
        #print("OUTPUT",outputs)
        #print("Labels",labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(epoch_loss/(i+1))
            
            
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j, inputs in enumerate(dataloader_test):
            labels, inputs = inputs['label'], inputs['face']
            img = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(img)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    # Construct log message
    log_message = f'Epoch {epoch + 1} train loss: {epoch_loss:.3f}, validation loss: {val_loss:.3f}'

    # Log the message
    logging.info(log_message)



    print('Epoch %d train loss: %.3f, validation loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader_train), val_loss / len(dataloader_test)))
        



# Save model
#epoch hangi epoch olduğu, 2.key olarak yazdığın şey, 3.sine optimazier state dict(değişen adımlar) 
torch.save(model.state_dict(), 'stgcn_model.pth')

#classes.csv 'yi okuyup filtereleyeceksin.
#scvleri pandasta okursan çok hızlı olur 
#main.py yapman lazım 
#main threadden başlat
