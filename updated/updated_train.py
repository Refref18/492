import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
import os
import logging
from son_loader import VideoPoseDataset
import csv
from son_loader import custom_collate_fn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'working')))
from net.st_gcn import Model
from net.utils.graph_mm import Graph

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_dir = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\mmpose-full'
info_file="D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\updated\\info.csv"
# Initialize dataset and data loader
graph=Graph(**{"layout": "mmpose", "strategy": "spatial"})
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
optimizer = optim.AdamW(model.parameters(), lr=0.001) #scheculear lr-> plateu lr tolerance azalmadıysa lr ı düşürüyor , ****multistep 0.0001 25 ve 45 yüzde 10 weight decay 10-5
print(len(dataloader_train))
# Open CSV file in write mode
with open('log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header row to CSV
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
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
            #print(img,labels)
            #print(len(inputs['face']))
            #print(inputs['label'], inputs['face'])
            # labels, inputs = inputs['label'], inputs['face']
            optimizer.zero_grad()
            # PYCM CONFUSION
            # cborn - heat map dpi artır
            # Initialize model
            outputs = model(img)
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

        # Split the log message to extract values
        epoch_num, train_loss, validation_loss = log_message.split()[1:]

        # Write values to CSV
        writer.writerow([epoch_num, train_loss, validation_loss])
        #logging.info(f'Epoch %d train loss: %.3f, validation loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader_train), val_loss / len(dataloader_test)))

        print('Epoch %d train loss: %.3f, validation loss: %.3f' % (epoch + 1, epoch_loss / len(dataloader_train), val_loss / len(dataloader_test)))
        



# Save model
#epoch hangi epoch olduğu, 2.key olarak yazdığın şey, 3.sine optimazier state dict(değişen adımlar) 
torch.save(model.state_dict(), 'stgcn_model.pth')

#classes.csv 'yi okuyup filtereleyeceksin.
#scvleri pandasta okursan çok hızlı olur 
#main.py yapman lazım 
#main threadden başlat
