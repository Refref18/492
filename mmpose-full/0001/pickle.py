import pickle

# Open the pickle file for reading
with open('User_2_001.pickle', 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)

# Do something with the data
print(data)
