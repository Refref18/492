import pickle

# Replace "path/to/file.pickle" with the actual path to your .pickle file
with open("D://2022-2023 2.dönem//Bitirme Projesi//face//492//mouth//User_2_004.pickle", "rb") as f:
    data = pickle.load(f)


# Print the type and contents of the loaded data
print(type(data))
print(data)
