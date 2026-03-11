import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ADE20KDataset(Dataset):
    """ADE20K Dataset for Semantic Segmentation."""
    
    def __init__(self, root, split="training", transform=None, target_transform=None):
        assert split in ["training", "validation"]
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        image_dir = os.path.join(root, "images", split)
        segmap_dir = os.path.join(root, "annotations", split)
        
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')
        ])
        self.segmap_paths = sorted([
            os.path.join(segmap_dir, f) for f in os.listdir(segmap_dir) if f.endswith('.png')
        ])

    def __len__(self):
        return len(self.image_paths)

    def load_segmentation_map_tensor(self, idx):
        segmap = Image.open(self.segmap_paths[idx])
        return torch.from_numpy(np.array(segmap)).long().unsqueeze(0)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        target = self.load_segmentation_map_tensor(idx)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        # Remove channel dimension for the mask -> (H, W)
        return image, target.squeeze(0)

def get_transforms(image_size=224):
    """Returns transformations for images and segmentation maps."""
    image_net_mean = [0.485, 0.456, 0.406]
    image_net_std = [0.229, 0.224, 0.225]
    
    image_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(image_net_mean, image_net_std)
    ])
    
    # Use NEAREST_EXACT to prevent interpolation of discrete class labels
    target_transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST_EXACT),
        transforms.CenterCrop(image_size)
    ])
    
    return image_transform, target_transform

