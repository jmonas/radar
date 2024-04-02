import pickle

# Specify the path to your pickle file
pickle_file_path = '/projects/FHEIDE/RADDet/train/gt/part_7/004872.pickle'

# Open the pickle file in binary read mode
try:
    with open(pickle_file_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)
        
        # Print the loaded data
        print(data)
except FileNotFoundError:
    print(f"The file {pickle_file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
