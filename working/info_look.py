import pickle
import os
import csv

full_path = "D:\\2022-2023 2.dönem\\Bitirme Projesi\\face\\492\\working\\"


def get_label(filename):
    # Load the CSV file
    with open((full_path+'classes.csv'), 'r' ,errors="ignore") as f:
        #print(csv.DictReader(f))
        reader = csv.DictReader(f)
        labels = {row['ClassID'].lstrip(
            '0'): row['ClassName_tr'] for row in reader}
    # Extract the ClassID from the filename
    class_id = filename.split('_')[-1].rstrip('.pickle').lstrip('0')
    print(class_id)
    print(labels)
    # Look up the label based on the ClassID
    if class_id in labels:
        label = labels[class_id]
    else:
        label = 'Unknown'

    return label

print(get_label(full_path+'User_2_004.pickle'))