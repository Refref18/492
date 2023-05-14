import os
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
import os
import sys
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'working')))
from update_datafilter import datafilter

#from working.pickle_filter import datafilter
#from working.label_all import get_label
import tensorflow as tf


class VideoPoseDataset(Dataset):
    def __init__(self, root_dir, info_file, train=True, transform=None, label_dict=None, index_to_label=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        
        #info_df = pd.read_csv(full_path+info_file)
        dtypes = {"RepeatID": str, "ClassID": str}
        info_df = pd.read_csv(info_file, dtype=dtypes)
        #print(info_df)
        label_dict = {label: i for i, label in enumerate(info_df['ClassID'].unique())}
        #print(label_dict)
        self.label_dict = label_dict
        self.index_to_label = {i: label for label, i in label_dict.items()}
        if self.train:
            label_dict = {label: i for i, label in enumerate(info_df['ClassID'].unique())}
            self.label_dict = label_dict
            self.index_to_label = {i: label for label, i in label_dict.items()}
            #print(self.label_dict)
            #print(self.index_to_label)
            self.info_df = info_df[~info_df['UserID'].str.endswith('4')]
        else:
            self.label_dict = label_dict
            self.index_to_label = index_to_label
            self.info_df = info_df[info_df['UserID'].str.endswith('4')]
        
        #print(label_dict)
        #print(self.index_to_label)
            
    def __len__(self):
        return len(self.info_df)
    
    def __getitem__(self, idx):
        row = self.info_df.iloc[idx]
        label = self.label_dict[row['ClassID']]
        #print(row)
        user, repeat = row['UserID'].split('_')[1], row['RepeatID']
        folder = os.path.join(self.root_dir, row['ClassID'])
        filename = f'User_{user}_{repeat}.pickle'
        """print(filename)
        video_data=datafilter(filename)
        print(video_data)"""
        file_path = os.path.join(folder, filename)
        # check if the file exists before attempting to read it
        #print(file_path)
        if os.path.exists(file_path):
            video_data=datafilter(file_path)
            #print(video_data)
            
            if self.transform:
                video_data = self.transform(video_data)

            return video_data, label
        """if self.transform:
            video_data = self.transform(video_data)"""
        #print(video_data)
        return video_data, label
def custom_collate_fn(batch):
    # Find the maximum sequence length
    #batch-> 4 tüm batchler
    #batch[0]->ilk batch lengthi 2: face ve label
    #batch[0][1] integer
    #len(batch[0][0]) .keys() face and label 
    #len(batch[0][0]['face']) 21
    #(batch[0][0]['label']) BOS
    max_length = max(
        len(sample[0]['face']['right_eyebrow_40']) for sample in batch)
    # print(batch[0]['label'])
    # print(max_length)
    keys_to_use = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
                   'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
                   'right_eye_59', 'right_eye_62', 'left_eye_65', 'left_eye_68',
                   'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89', 'right_eye_60', 'right_eye_63', 'left_eye_66', 'left_eye_69']

    # Initialize the tensors for the padded batch
    padded_batch = {'face': {key: [] for key in keys_to_use}}
    for i in range(max_length):
        for key in keys_to_use:
            padded_batch['face'][key].append([])

    # Fill the tensors with the data from the batch
    for sample in batch:
        for key in keys_to_use:
            #print(key)
            # Get the data for this key
            data = sample[0]['face'][key]
            # print(type(data[0]))
            # Pad the data with zeros if necessary
            if len(data) < max_length:
                # print(len(data))
                num_zeros = max_length - len(data)
                zero_padding = np.zeros((num_zeros, 3))
                # print("zero:",zero_padding)
                # print(data)
                data = np.vstack([data, zero_padding])
                # print("npstack ",data)
            # print(len(padded_batch['face'][key]),len(data))

            # Add the data to the tensor
            padded_batch['face'][key] = data

    # Convert the numpy arrays to PyTorch tensors
    for key in keys_to_use:
        padded_batch['face'][key] = torch.from_numpy(
            padded_batch['face'][key])
         # print(padded_batch)

    # Convert the labels to PyTorch tensors
    # print([sample['label'] for sample in batch])
    # padded_batch['label'] = torch.tensor([sample['label'] for sample in batch])
    # padded_batch['label'] = [(sample['label']) for sample in batch]
    # padded_batch['label'] = torch.tensor([sample['label'] for sample in batch])
    #padded_batch['label'] = tf.constant([sample['label'] for sample in batch])
    padded_batch['label'] = torch.LongTensor([sample[1] for sample in batch]) #integera çevrirmek
    #print(padded_batch['label'])

    return padded_batch


"""if __name__ == '__main__':
    data_dir = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\mmpose-full'
    dataset = VideoPoseDataset(data_dir,train=True,info_file="info.csv")
    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=custom_collate_fn)
    # print(dataloader)

    for poses in dataloader:
        print(""),"""


"""# create the train set with new label_dict and index_to_label dictionaries
train_dataset = VideoPoseDataset(root_dir, train_info_file, train=True, transform=train_transform)

# create the test set with the same label_dict and index_to_label dictionaries as the train set
test_dataset = VideoPoseDataset(root_dir, test_info_file, train=False, transform=test_transform, label_dict=train_dataset.label_dict, index_to_label=train_dataset.index_to_label)
"""