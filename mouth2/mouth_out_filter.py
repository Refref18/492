import numpy as np
import os
import glob
import pickle
# Define the list of keys we want to keep


def datafilter(filename):
    keys_to_keep = ['mouth_71', 'mouth_72', 'mouth_73', 'mouth_74', 'mouth_75', 'mouth_76', 'mouth_77', 'mouth_78',
                    'mouth_79', 'mouth_80', 'mouth_81', 'mouth_82']

    # Create a new dictionary to store the filtered data

    filtered_data = {'face': {}, 'label': {},
                     'pose': {}, 'hand_left': {}, 'hand_right': {}}
    # print(labels)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    for key, value in data['face'].items():
        if key in keys_to_keep:
            filtered_data['face'][key] = value
    active_frame_keys_pose = ['left_hip', 'right_hip', 'nose']
    for key, value in data['pose'].items():
        if key in active_frame_keys_pose:
            filtered_data['pose'][key] = value
    filtered_data['hand_left']['left_lunate_bone'] = data['hand_left']['left_lunate_bone']
    filtered_data['hand_right']['right_lunate_bone'] = data['hand_right']['right_lunate_bone']

    """
    print(type(filtered_data))
    print(type(filtered_data['label']))
    print(type(filtered_data['face']))
    print(type(filtered_data['face']['right_eyebrow_40']))
    print(type(filtered_data['face']['right_eyebrow_40'][0]))
    """
    # print("filtered_data", filtered_data['label'])
    return (filtered_data)


# datafilter('User_2_004.pickle')
