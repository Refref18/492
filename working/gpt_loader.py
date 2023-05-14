import os
import glob
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from net.utils.graph_mm import Graph
from pickle_filter import datafilter
from label_all import get_label
import tensorflow as tf


class VideoPoseDataset(Dataset):
    
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.video_files = glob.glob(os.path.join(data_dir, '*.pickle'))
        
        # Split the video files into train and test sets based on the user
        users = [int(os.path.basename(video_file).split("_")[0]) for video_file in self.video_files]
        user_ids = sorted(list(set(users)))
        user_id_split = int(len(user_ids) * split_ratio)
        train_user_ids = user_ids[:user_id_split] if train else user_ids[user_id_split:]
        train_video_files = [self.video_files[i] for i, user in enumerate(users) if user in train_user_ids]
        test_video_files = [self.video_files[i] for i, user in enumerate(users) if user not in train_user_ids]
        self.video_files = train_video_files if train else test_video_files
        
        # TODO: Define the vocabulary label list
        
        # Initialize gloss-to-index and index-to-gloss dictionaries
        if train:
            all_glosses = set()
            for video_file in self.video_files:
                video_data = datafilter(video_file)
                glosses = set(get_label(video_data))
                all_glosses = all_glosses.union(glosses)
            self.gloss2idx = {gloss: idx for idx, gloss in enumerate(sorted(list(all_glosses)))}
            self.indextogloss = {idx: gloss for gloss, idx in self.gloss2idx.items()}
            self.num_classes = len(self.gloss2idx)
        else:
            # Use the same dictionaries as in the training set
            self.gloss2idx = gloss2idx_train
            self.indextogloss = indextogloss_train
            self.num_classes = num_classes_train
        
        # Load the data
        # TODO: Implement lazy loading or other memory optimization techniques
        self.data = []
        for video_file in self.video_files:
            video_data = datafilter(video_file)
            label = get_label(video_data)
            label_idx = [self.gloss2idx[gloss] for gloss in label]
            self.data.append({'video': video_data, 'label': label_idx})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        video_data = sample['video']
        label = sample['label']
        # adj = self.adj_matrix
        # return video_data,adj
        return {'face': video_data, 'label': label}


def custom_collate_fn(batch):
    # Find the maximum sequence length
    max_length = max(
        len(sample['face']['right_eyebrow_40']) for sample in batch)

    keys_to_use = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
                   'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
                   'right_eye_59', 'right_eye_62
