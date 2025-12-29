import torch
import numpy as np
import torchvision
from torchvision.io import read_image
import os

def scale_bands(img,satellite="landsat"):
    """Scale bands to 0-1"""
    img = img.astype("float32")
    if satellite == "landsat":
        img = np.clip(img * 0.0000275 - 0.2, 0, 1)
    elif satellite == "sentinel":
        img = np.clip(img/10000, 0, 1)
    return img

# Classes
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        """Initialise the dataset
        Args:
        paths (list): List of file paths
        target_pos (int): Position of the target band in the input data
        incl_bands (list): List of bands to include in the input data
        satellite (str): Satellite type"""

        self.paths = paths

        # Define dataset parameters
        self.target = 9
        self.incl_bands = [0, 1, 2, 3, 4, 5, 6]
        self.satellite = "landsat"

    def __getitem__(self, idx):
        """Get image and binary mask for a given index"""

        path = self.paths[idx]
        instance = np.load(path)

        bands = instance[:, :, self.incl_bands]  # Only include specified bands
        bands = bands.astype(np.float32) 

        # Normalise bands
        bands = scale_bands(bands, self.satellite)

        # Convert to tensor
        bands = bands.transpose(2, 0, 1)
        bands = torch.tensor(bands, dtype=torch.float32)

        # Get target
        mask_1 = instance[:, :, self.target].astype(np.int8)  # Water = 1, Land = 0
        mask_1[np.where(mask_1 == -1)] = 0  # Set nodata values to 0
        mask_0 = 1 - mask_1

        target = np.array([mask_0, mask_1])
        target = torch.Tensor(target).squeeze()

        return bands, target

    def __len__(self):
        return len(self.paths)
    
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, class_names=None, num_classes=None):
        self.paths = paths
        if class_names is not None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        else:
            self.class_names = None
            self.num_classes = num_classes  # fallback

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = read_image(path).float() / 255.0  # Normalize image
        filename = os.path.basename(path)  # e.g., "greg.jpg"

        if self.class_names:  # string label mode
            class_name = filename.split(".")[0]  # "greg"
            target_label = self.class_names.index(class_name)
        else:  # numeric mode
            target_label = int(filename.split("_")[0])  

        # One-hot encode
        target = torch.zeros(self.num_classes)
        target[target_label] = 1

        return image, target

    def __len__(self):
        return len(self.paths)