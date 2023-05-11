import pickle
import os
import csv

full_path = "C:\\Users\\eatron1\\Desktop\\termproject-main\\termproject-main\\working\\"

def label_directory(directory):
    # Load the CSV file
    with open('classes.csv', 'r') as f:
        reader = csv.DictReader(f)
        labels = {row['ClassID']: row['ClassName_tr'] for row in reader}
    #print(labels)

    # Loop through each folder in the directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Check if the folder is a directory and not a file
        if os.path.isdir(folder_path):
            # Get the ClassID from the folder name
            class_id = folder
            #print(class_id)

            # Look up the label based on the ClassID
            if class_id in labels:
                label = labels[class_id]
            else:
                label = 'Unknown'

            # Loop through each file in the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # Print the label and filename
                print(f'{label}: {filename}')




def get_label(filename):
    # Load the CSV file
    with open((full_path+'classes.csv'), 'r' ,errors="ignore") as f:
        print(csv.DictReader(f))
        reader = csv.DictReader(f)
        labels = {row['ClassID'].lstrip(
            '0'): row['ClassName_tr'] for row in reader}
    # Extract the ClassID from the filename
    class_id = filename.split('_')[-1].rstrip('.pickle').lstrip('0')
    class_id=int(class_id)
    # Look up the label based on the ClassID
    if class_id in labels:
        label = labels[class_id]
    else:
        label = 'Unknown'

    return label

#print(get_label('C:\\Users\\eatron1\\Desktop\\termproject-main\\termproject-main\\working\\User_2_004.pickle'))
