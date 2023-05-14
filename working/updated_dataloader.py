import os
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from net.utils.graph_mm import Graph
from pickle_filter import datafilter
import tensorflow as tf


class VideoPoseDataset(Dataset):
    
    def __init__(self, data_dir):
        #DONE train mi test mi
        #user 4 mesela test
        #22bin satırlık şeyde hangi videolar test hangi videolar train
        #o dosyayı bir kere okuycam
        #split method csv method olarak yaz bunu usera göre böl!
        #user dependancy 
        #vocabulary label listesi 
        #indexli enumerate etmen laızm, 1 tane dicionary 
        #gloss to index {0010: class_iddsinden enumeratte gelen}
        #num_classes dışardan kullanmak için
        #gloss2idx ve indextogloss testte de oluşturma traindekini kullan 
        #gloss2idx= {gloss: idx for idx, gloss in enumarate(uniques_glosses)} -replike şeyler çıkmasın
        #indextogloss ={class_id:0010} hangi index hangi labela denk geliyor
        #dataları komple load? -SONRA
        self.data_dir = data_dir
        self.video_files = glob.glob(os.path.join(data_dir, '*.pickle'))
        # TODO CHANGE THE DIRECTORY !!

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_data = datafilter(video_file)
        print(video_data)
        """with open(video_file, 'rb') as f:
            video_data = pickle.load(f)"""
        # adj = self.adj_matrix
        # return video_data,adj
        return video_data


def custom_collate_fn(batch):
    # Find the maximum sequence length
    max_length = max(
        len(sample['face']['right_eyebrow_40']) for sample in batch)
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
            # Get the data for this key
            data = sample['face'][key]
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
    padded_batch['label'] = torch.LongTensor([sample['label'] for sample in batch]) #integera çevrirmek

    return padded_batch

"""
if __name__ == '__main__':
    data_dir = 'D:/2022-2023 2.dönem/Bitirme Projesi/face'
    dataset = VideoPoseDataset(data_dir)
    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=True, collate_fn=custom_collate_fn)
    # print(dataloader)

    for poses in dataloader:
        print("")
"""
