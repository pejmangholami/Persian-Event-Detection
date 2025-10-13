import numpy as np

# Load the .npy file
try:
    data = np.load('Language-Model_Scoring/AllData.npy', allow_pickle=True)
    print("File loaded successfully.")

    # Print the type of the loaded data
    print(f"Type of data: {type(data)}")

    # If the data is a numpy array, print its shape
    if isinstance(data, np.ndarray):
        print(f"Shape of the array: {data.shape}")

        # Print the first 5 elements to inspect the structure
        print("First 5 elements:")
        for i, item in enumerate(data[:5]):
            print(f"Element {i}: {item}")

except FileNotFoundError:
    print("Error: The file 'Language-Model_Scoring/AllData.npy' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")