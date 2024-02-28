import torch
import numpy as np

def convert_pt_to_npz(pt_file_path, npz_file_path):
    """
    Converts a PyTorch model or state dictionary (.pt) to a NumPy Zip archive (.npz) with checks to ensure only tensors are converted.

    Parameters:
        pt_file_path (str): The file path to the .pt file.
        npz_file_path (str): The desired output file path for the .npz file.
    """
    # Load the PyTorch model or state dictionary
    model_state_dict = torch.load(pt_file_path)

    # Initialize an empty dictionary to store NumPy arrays
    arrays_dict = {}

    # Iterate through the state dictionary
    for k, v in model_state_dict.items():
        # Check if the item is a tensor
        if torch.is_tensor(v):
            # Convert tensor to NumPy array after moving it to CPU
            arrays_dict[k] = v.cpu().numpy()
        else:
            # Directly store the item if it's not a tensor (e.g., metadata)
            arrays_dict[k] = v

    # Save the arrays to an .npz file
    np.savez(npz_file_path, **arrays_dict)

if __name__ == "__main__":
    pt_file_path = 'ModelTest1.pt'  # Path to your .pt file
    npz_file_path = 'ModelTest1.npz'  # Desired output path
    convert_pt_to_npz(pt_file_path, npz_file_path)