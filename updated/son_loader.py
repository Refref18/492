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
    def __init__(self, root_dir, info_file, train=True, transform=None, label_dict=None, index_to_label=None,unique_nodes=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.unique_nodes=unique_nodes
        
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
        #if os.path.exists(file_path):#bunu koymaman lazım 
        video_data=datafilter(file_path)
        #print(video_data)
        #video_data=self.process_skeleton(video_data,self.unique_nodes)
        #print(video_data.shape)
        if self.transform:
            video_data = self.transform(video_data)

        return video_data, label
        """if self.transform:
            video_data = self.transform(video_data)"""
        #print(video_data)
        #return video_data, label

# Remove dominant movements where signer raises and drops down their hands
def get_active_frames( input_raw):
    threshold = ((((input_raw['pose']['left_hip'][:, 1] + input_raw['pose']['right_hip'][:, 1]) / 2) * 7) +
                 input_raw['pose']['nose'][:, 1]) / 10

    active_frames = np.minimum(input_raw['hand_left']['left_lunate_bone'][:, 1],
                               input_raw['hand_right']['right_lunate_bone'][:, 1]) < threshold

    active_frame_indices = np.argwhere(active_frames).squeeze()
    return active_frame_indices

def process_hands(input_raw):
    keys = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
            'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
            'right_eye_59', 'right_eye_60', 'right_eye_62', 'right_eye_63', 'left_eye_65', 'left_eye_66', 'left_eye_68', 'left_eye_69',
            'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89']
    active_frame_indices = get_active_frames(input_raw)
    active_frame_indices = active_frame_indices if active_frame_indices.size > 10 else np.arange(
        0, len(input_raw))
    input_raw = {**input_raw['face']}
    
    input = np.array([input_raw[jn] for jn in keys]).transpose(1, 0, 2)
    
    input = input[active_frame_indices, ...].transpose(1, 0, 2)
    a={"face": input}
    return a
    

def process_skeleton(input_raw, nodes):
    # print(input_raw)
    keys = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
            'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
            'right_eye_59', 'right_eye_60', 'right_eye_62', 'right_eye_63', 'left_eye_65', 'left_eye_66', 'left_eye_68', 'left_eye_69',
            'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89']
    
    """active_frame_indices = get_active_frames(input_raw)
    active_frame_indices = active_frame_indices if active_frame_indices.size > 10 else np.arange(
        0, len(input))"""
    
    
    input_raw = {**input_raw['face']}
    input = np.array([input_raw[jn] for jn in range(21)]).transpose(2, 1, 0)

    """input = input[active_frame_indices, ...]
    input = input.transpose(2, 0, 1)"""
    
    
    input = np.expand_dims(input, axis=-1)
   
    
    #active frameleri bul

    
    # Convert dictionary values to arrays and apply transpose
    #input = np.array([input_raw[jn] for jn in keys]).transpose(2, 1, 0)
    
    # .transpose(1,0,2)
    #print(input.shape)
    return input

def custom_collate_fn(batch):
    # Find the maximum sequence length
    #batch-> 4 tüm batchler
    #batch[0]->ilk batch lengthi 2: face ve label
    #batch[0][1] integer
    #len(batch[0][0]) .keys() face and label 
    #len(batch[0][0]['face']) 21
    #(batch[0][0]['label']) BOS
    
    #-----
    #len(batch) number of batches
    #batch[0][0] the nodes
    #print(batch[0][0].shape)
    #print(batch[0][0])
    max_length = max(
        len(sample[0]['face']['right_eyebrow_40']) for sample in batch)
    
    print(max_length)
    for i,sample in enumerate(batch):
        a = process_hands(sample[0])
        batch[i][0]['face'] = a['face']
        
    
    max_length = max(
        len(sample[0]['face'][0]) for sample in batch)
    print(max_length)
    # print(batch[0]['label'])
    # print(max_length)
    keys_to_use = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
                   'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
                   'right_eye_59', 'right_eye_62', 'left_eye_65', 'left_eye_68',
                   'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89', 'right_eye_60', 'right_eye_63', 'left_eye_66', 'left_eye_69']

    # Initialize the tensors for the padded batch
    padded_batch_list = [
        {'face': {key: [None] * max_length for key in range(21)}} for _ in range(len(batch))]
    for i in range(len(batch)):
        padded_batch_list[i]['pose']=batch[i][0]['pose']
        padded_batch_list[i]['hand_left'] = batch[i][0]['hand_left']
        padded_batch_list[i]['hand_right'] = batch[i][0]['hand_right']
        
    
    a={}

    #padded_batch = {'face': {key: [] for key in (keys_to_use)}}
    """for i in range(max_length):
        for key in (keys_to_use):
            padded_batch['face'][key].append([])"""

    # Fill the tensors with the data from the batch
    for i,sample in enumerate(batch):
        for key in range(21):
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
            #print("A",padded_batch_list[i]['face'][key])
            padded_batch_list[i]['face'][key] = data
            #print("B",padded_batch_list[i]['face'][key])
            
        padded_batch_list[i] = process_skeleton(
            padded_batch_list[i], [])
        #print(padded_batch_list[i])

    # Convert the numpy arrays to PyTorch tensors
    """for padded_batch in padded_batch_list:
        for key in (keys_to_use):
            padded_batch[0][key] = torch.from_numpy(
                padded_batch[0][key])
            # print(padded_batch)"""
    
    padded_batch_list_tensor = torch.stack(
        [torch.from_numpy(arr) for arr in padded_batch_list], dim=0)
    
    a['face']=padded_batch_list_tensor
    # torch.tensor(img, dtype=torch.float32) folat.tensor

    # Convert the labels to PyTorch tensors
    # print([sample['label'] for sample in batch])
    # padded_batch['label'] = torch.tensor([sample['label'] for sample in batch])
    # padded_batch['label'] = [(sample['label']) for sample in batch]
    # padded_batch['label'] = torch.tensor([sample['label'] for sample in batch])
    #padded_batch['label'] = tf.constant([sample['label'] for sample in batch])
    a['label'] = torch.LongTensor(
        [sample[1] for sample in batch])  # integera çevrirmek
    #print(padded_batch['label'])
    #print(padded_batch)
    """Shape:
        - Input: : math: `(N, in_channels, T_{ in }, V_{ in }, M_{ in })`
        - Output: : math: `(N, num_class)` where
        : math: `N` is a batch size,
            : math: `T_{ in }` is a length of input sequence,
            : math: `V_{ in }` is the number of graph nodes,
            : math: `M_{ in }` is the number of instance in a frame.
            
    batch_size*3*T*V*M"""

    return a


"""if __name__ == '__main__':
    data_dir = 'D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\mmpose-full'
    dataset = VideoPoseDataset(data_dir,train=True,info_file="info.csv")
    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, collate_fn=custom_collate_fn)
    # print(dataloader)

    for poses in dataloader:
        print("")"""


"""# create the train set with new label_dict and index_to_label dictionaries
train_dataset = VideoPoseDataset(root_dir, train_info_file, train=True, transform=train_transform)

# create the test set with the same label_dict and index_to_label dictionaries as the train set
test_dataset = VideoPoseDataset(root_dir, test_info_file, train=False, transform=test_transform, label_dict=train_dataset.label_dict, index_to_label=train_dataset.index_to_label)
"""