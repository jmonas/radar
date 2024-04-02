import pickle
import sys
# Specify the path to your pickle file
# pickle_file_path = '/projects/FHEIDE/RADDet/train/gt/part_7/004872.pickle'


# Check if the user has provided the file name as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python3 read_single_gt.py <pickle_file_path>")
    sys.exit(1)

# The first command-line argument after the script name is the path to the pickle file
pickle_file_path = sys.argv[1]

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
