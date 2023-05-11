import numpy as np
import os
import glob
import pickle
from label_all import get_label
# Define the list of keys we want to keep


def datafilter(filename):
    keys_to_keep = ['right_eyebrow_40', 'right_eyebrow_42', 'right_eyebrow_44', 'left_eyebrow_45',
                    'left_eyebrow_47', 'left_eyebrow_49', 'nose_54', 'nose_56', 'nose_58',
                    'right_eye_59', 'right_eye_62', 'left_eye_65', 'left_eye_68',
                    'mouth_83', 'mouth_85', 'mouth_87', 'mouth_89']

    # Define the keys for which we want to take the average of adjacent points
    keys_to_average = {'right_eye_60': ['right_eye_59', 'right_eye_62'],
                       'right_eye_63': ['right_eye_62', 'right_eye_64'],
                       'left_eye_66': ['left_eye_65', 'left_eye_68'],
                       'left_eye_69': ['left_eye_68', 'left_eye_70']}

    # Create a new dictionary to store the filtered data

    filtered_data = {'face': {}, 'label': {}}
    labels = get_label(filename)
    #print(labels)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    for key, value in data['face'].items():
        if key in keys_to_keep:
            filtered_data['face'][key] = value
        elif key in keys_to_average:
            x_vals = []
            y_vals = []
            prob_vals = []
            # print(value)
            for k in value:
                # print(k[0])
                # print(value)
                x_vals.append(k[0])
                y_vals.append(k[1])
                prob_vals.append(k[2])
            avg_x = np.mean(x_vals)
            avg_y = np.mean(y_vals)
            avg_prob = np.mean(prob_vals)
            
            filtered_data['face'][key] = np.array([
                [avg_x, avg_y, avg_prob]] * len(value))
    filtered_data['label'] = labels
    """
    print(type(filtered_data))
    print(type(filtered_data['label']))
    print(type(filtered_data['face']))
    print(type(filtered_data['face']['right_eyebrow_40']))
    print(type(filtered_data['face']['right_eyebrow_40'][0]))
    """
    #print("filtered_data", filtered_data['label'])
    return (filtered_data)


#datafilter('User_2_004.pickle')
