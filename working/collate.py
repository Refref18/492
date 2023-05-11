import torch
import numpy as np

def custom_collate_fn(batch):
    # Find the maximum sequence length
    max_length = max(len(sample['data']['face']['right_eyebrow_40']) for sample in batch)
    keys_to_use = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
                   'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
                   'right_eye_59', 'right_eye_62', 'left_eye_65', 'left_eye_68',
                   'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89', 'right_eye_60', 'right_eye_63', 'left_eye_66', 'left_eye_69']
    
    # Initialize the tensors for the padded batch
    padded_batch = {'data': {'face': {key: [] for key in keys_to_use}}}
    for i in range(max_length):
        for key in keys_to_use:
            padded_batch['data']['face'][key].append([])
    
    # Fill the tensors with the data from the batch
    for sample in batch:
        for key in keys_to_use:
            # Get the data for this key
            data = sample['data']['face'][key]
            
            # Pad the data with zeros if necessary
            if len(data) < max_length:
                num_zeros = max_length - len(data)
                zero_padding = np.zeros((num_zeros, 3))
                data = np.vstack([data, zero_padding])
            
            # Add the data to the tensor
            padded_batch['data']['face'][key] = np.vstack([padded_batch['data']['face'][key], data])
    
    # Convert the numpy arrays to PyTorch tensors
    for key in keys_to_use:
        padded_batch['data']['face'][key] = torch.from_numpy(padded_batch['data']['face'][key])
    
    # Convert the labels to PyTorch tensors
    padded_batch['label'] = torch.tensor([sample['label'] for sample in batch])
    
    return padded_batch
